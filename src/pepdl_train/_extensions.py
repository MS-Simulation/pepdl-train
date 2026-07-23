"""pepdl-train extensions — re-attach the training methods (fine_tune*) that were extracted out of the
pepdl INFERENCE package (so inference imports nothing training). Importing pepdl_train monkeypatches
them back onto the predictor classes; the bodies lazy-import torch/datasets/sagepy so attaching is
cheap. Provenance: method bodies verbatim from imspy-predictors (self-bound, pepdl.losses->pepdl_train.losses)."""
from __future__ import annotations

from pepdl.ccs.predictors import DeepPeptideIonMobilityApex
from pepdl.rt.predictors import DeepChromatographyApex
from pepdl.intensity.predictors import DeepPeptideIntensityPredictor

def _DeepPeptideIonMobilityApex_fine_tune_model(
    self,
    data: pd.DataFrame,
    batch_size: int = 64,
    epochs: int = 150,
    learning_rate: float = 1e-3,
    patience: int = 6,
    verbose: bool = False,
    decoys_separate: bool = True,
) -> None:
    """
    Fine-tune the model on new data.

    Args:
        data: DataFrame with columns: sequence, charge, calcmass, ims
        batch_size: Training batch size
        epochs: Maximum number of epochs
        learning_rate: Learning rate
        patience: Early stopping patience
        verbose: Whether to print progress
        decoys_separate: Whether to handle decoys separately
    """
    from torch.utils.data import DataLoader, TensorDataset

    assert 'sequence' in data.columns, 'Data must contain column "sequence"'
    assert 'charge' in data.columns, 'Data must contain column "charge"'
    assert 'calcmass' in data.columns, 'Data must contain column "calcmass"'
    assert 'ims' in data.columns, 'Data must contain column "ims"'

    mz = [
        calculate_mz(m, z)
        for m, z in zip(
            data.calcmass.values,
            data.charge.values.astype(np.int64),
        )
    ]
    charges = data.charge.values.astype(np.int64)

    if decoys_separate:
        sequences = []
        for _, row in data.iterrows():
            if not row.decoy:
                sequences.append(row.sequence_modified)
            else:
                sequences.append(row.sequence_decoy_modified)
    else:
        sequences = list(data.sequence_modified.values)

    # Drop training rows whose charge is outside the model's
    # one-hot domain (1..4). They cannot be one-hot encoded without
    # tripping F.one_hot's CUDA index assertion. Charges of 5+ leak
    # through sage matching at ~0.15% on HeLa-class data even when
    # precursor_charge is configured as [2, 4].
    n_total = len(sequences)
    valid_mask = (charges >= 1) & (charges <= 4)
    n_invalid = int((~valid_mask).sum())
    if n_invalid:
        warnings.warn(
            f"fine_tune_model: dropping {n_invalid} of {n_total} training "
            f"PSMs with charge outside [1, 4].",
            RuntimeWarning,
            stacklevel=2,
        )
        sequences = [s for s, m in zip(sequences, valid_mask) if m]
        mz = [m for m, ok in zip(mz, valid_mask) if ok]
        charges = charges[valid_mask]
        inv_mob = data.ims.values[valid_mask]
    else:
        inv_mob = data.ims.values

    if len(sequences) == 0:
        warnings.warn(
            "fine_tune_model: no PSMs in valid charge range; skipping fine-tune.",
            RuntimeWarning,
            stacklevel=2,
        )
        self._finetune_history = {"epochs": [], "train_loss": [], "val_loss": []}
        return

    ccs = np.array([
        one_over_k0_to_ccs(i, m, z)
        for i, m, z in zip(inv_mob, mz, charges)
    ])

    # Prepare data
    tokens = self._preprocess_sequences(sequences)
    mz_tensor = torch.tensor(mz, dtype=torch.float32, device=self._device)
    charges_onehot = F.one_hot(
        torch.tensor(charges, dtype=torch.long, device=self._device) - 1,
        num_classes=4,
    ).float()
    ccs_tensor = torch.tensor(ccs, dtype=torch.float32, device=self._device).unsqueeze(1)

    # Group-aware (peptide × charge) split: same (modseq, charge) → same
    # fold. PSM-level random split would leak — the predictor is
    # deterministic per (sequence, charge) so identical inputs in
    # train and val collapse val loss to the instrument's IM-noise
    # floor, not the model's generalization.
    n = len(sequences)
    group_keys = np.array([f"{s}_{int(c)}" for s, c in zip(sequences, charges)])
    uniq, inv = np.unique(group_keys, return_inverse=True)
    n_groups = len(uniq)
    n_val_groups = max(1, int(n_groups * 0.2))
    if n_val_groups >= n_groups:
        n_val_groups = n_groups - 1
    rng_np = np.random.default_rng(42)
    perm_groups = rng_np.permutation(n_groups)
    val_groups = set(perm_groups[:n_val_groups].tolist())
    mask_val = np.fromiter((g in val_groups for g in inv),
                               dtype=bool, count=n)
    val_idx   = torch.from_numpy(np.flatnonzero(mask_val))
    train_idx = torch.from_numpy(np.flatnonzero(~mask_val))
    if verbose:
        print(f"[im-ft] {n} PSMs ({n_groups:,} unique (modseq,charge)) → "
              f"train {len(train_idx):,}, val {len(val_idx):,} "
              f"(val groups: {n_val_groups:,})")

    # Create datasets
    train_dataset = TensorDataset(
        mz_tensor[train_idx],
        charges_onehot[train_idx],
        tokens[train_idx],
        ccs_tensor[train_idx],
    )
    val_dataset = TensorDataset(
        mz_tensor[val_idx],
        charges_onehot[val_idx],
        tokens[val_idx],
        ccs_tensor[val_idx],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Training setup
    self.model.train()
    optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, min_lr=1e-6
    )
    checkpoint = InMemoryCheckpoint(patience=patience)

    history = {"epochs": [], "train_loss": [], "val_loss": []}
    for epoch in range(epochs):
        # Training
        self.model.train()
        train_loss = 0
        for batch in train_loader:
            mz_b, charge_b, tokens_b, ccs_b = batch
            optimizer.zero_grad()

            # Handle different model types
            if hasattr(self.model, 'predict_ccs'):
                pred = self.model.predict_ccs(
                    tokens_b,
                    mz_b,
                    torch.argmax(charge_b, dim=1) + 1,
                )
            else:
                pred = self.model(mz_b, charge_b, tokens_b)
            pred = _extract_prediction_mean(pred, "ccs")

            loss = F.l1_loss(pred, ccs_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        # Validation
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                mz_b, charge_b, tokens_b, ccs_b = batch

                if hasattr(self.model, 'predict_ccs'):
                    pred = self.model.predict_ccs(
                        tokens_b,
                        mz_b,
                        torch.argmax(charge_b, dim=1) + 1,
                    )
                else:
                    pred = self.model(mz_b, charge_b, tokens_b)
                pred = _extract_prediction_mean(pred, "ccs")

                val_loss += F.l1_loss(pred, ccs_b).item()

        val_loss /= max(len(val_loader), 1)
        scheduler.step(val_loss)

        history["epochs"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        # Print every 5 epochs (was 10) — matches RT/intensity pattern so
        # the report's per-head FT plot has comparable point density.
        if verbose and epoch % 5 == 0:
            print(f"Epoch {epoch}: im train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f}")

        if checkpoint.step(val_loss, self.model):
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

    checkpoint.restore(self.model)
    self.model.eval()
    self._finetune_history = history
DeepPeptideIonMobilityApex.fine_tune_model = _DeepPeptideIonMobilityApex_fine_tune_model

def _DeepChromatographyApex_fine_tune_model(
    self,
    data: pd.DataFrame,
    batch_size: int = 64,
    epochs: int = 150,
    learning_rate: float = 1e-3,
    patience: int = 6,
    verbose: bool = False,
    decoys_separate: bool = True,
) -> None:
    """
    Fine-tune the model on new data.

    Args:
        data: DataFrame with columns: sequence, retention_time_projected
        batch_size: Training batch size
        epochs: Maximum number of epochs
        learning_rate: Learning rate
        patience: Early stopping patience
        verbose: Whether to print progress
        decoys_separate: Whether to handle decoys separately
    """
    assert 'sequence' in data.columns, 'Data must contain column "sequence"'
    assert 'retention_time_projected' in data.columns, 'Data must contain column "retention_time_projected"'

    if decoys_separate:
        sequences = []
        for _, row in data.iterrows():
            if not row.decoy:
                sequences.append(row.sequence)
            else:
                sequences.append(row.sequence_decoy_modified)
    else:
        sequences = list(data.sequence.values)

    rts = data.retention_time_projected.values
    self._fine_tune(sequences, rts, batch_size, epochs, learning_rate, patience, verbose)
DeepChromatographyApex.fine_tune_model = _DeepChromatographyApex_fine_tune_model

def _DeepChromatographyApex__fine_tune(
    self,
    sequences: List[str],
    rts: NDArray,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    patience: int,
    verbose: bool,
) -> None:
    """PyTorch fine-tuning."""
    from torch.utils.data import DataLoader, TensorDataset

    tokens = self._preprocess_sequences(sequences)
    rt_tensor = torch.tensor(rts, dtype=torch.float32, device=self._device).unsqueeze(1)

    n = len(sequences)
    n_train = int(0.8 * n)
    indices = torch.randperm(n)

    train_dataset = TensorDataset(tokens[indices[:n_train]], rt_tensor[indices[:n_train]])
    val_dataset = TensorDataset(tokens[indices[n_train:]], rt_tensor[indices[n_train:]])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    self.model.train()
    optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, min_lr=1e-6)
    checkpoint = InMemoryCheckpoint(patience=patience)

    history = {"epochs": [], "train_loss": [], "val_loss": []}
    for epoch in range(epochs):
        self.model.train()
        train_loss = 0.0
        for tokens_b, rt_b in train_loader:
            optimizer.zero_grad()
            pred = _extract_prediction_mean(self.model(tokens_b), "rt")
            loss = F.l1_loss(pred, rt_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for tokens_b, rt_b in val_loader:
                pred = _extract_prediction_mean(self.model(tokens_b), "rt")
                val_loss += F.l1_loss(pred, rt_b).item()
        val_loss /= max(len(val_loader), 1)
        scheduler.step(val_loss)

        history["epochs"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if checkpoint.step(val_loss, self.model):
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

    self._finetune_history = history

    checkpoint.restore(self.model)
    self.model.eval()
DeepChromatographyApex._fine_tune = _DeepChromatographyApex__fine_tune

def _DeepPeptideIntensityPredictor_fine_tune_model(
    self,
    data: pd.DataFrame,
    batch_size: int = 64,
    epochs: int = 50,
    learning_rate: float = 1e-4,
    patience: int = 5,
    divide_collision_energy_by: float = 1e2,
    verbose: bool = False,
) -> None:
    """
    Fine-tune the native intensity model on observed 174-vector targets.

    Args:
        data: DataFrame with columns: sequence, charge, collision_energy,
            intensity_target. The target must use the native ordinal-major
            174-vector layout and mark impossible ions as -1.
        batch_size: Training batch size
        epochs: Maximum number of epochs
        learning_rate: Learning rate
        patience: Early stopping patience
        divide_collision_energy_by: CE normalization factor
        verbose: Whether to print progress
    """
    assert 'sequence' in data.columns, 'Data must contain column "sequence"'
    assert 'charge' in data.columns, 'Data must contain column "charge"'
    assert 'collision_energy' in data.columns, 'Data must contain column "collision_energy"'
    assert 'intensity_target' in data.columns, 'Data must contain column "intensity_target"'

    from torch.utils.data import DataLoader, TensorDataset
    from pepdl_train.losses import masked_spectral_distance

    if len(data) < 2:
        if verbose:
            print("Skipping intensity fine-tune: need at least two PSMs")
        return

    sequences = data.sequence.tolist()
    charges = data.charge.astype(np.int64).tolist()
    collision_energies = (data.collision_energy.astype(float) / divide_collision_energy_by).tolist()
    targets = np.vstack(data.intensity_target.to_numpy()).astype(np.float32)
    if targets.shape != (len(data), 174):
        raise ValueError(f"intensity_target must have shape (n, 174), got {targets.shape}")

    tokens, charge_tensor, ce_tensor = self._preprocess(
        sequences,
        charges,
        collision_energies,
    )
    tokens = tokens.to(self._device)
    charge_tensor = charge_tensor.to(self._device)
    ce_tensor = ce_tensor.to(self._device)
    target_tensor = self._torch.tensor(targets, dtype=self._torch.float32, device=self._device)

    n = len(sequences)
    n_train = max(1, int(0.8 * n))
    if n_train >= n:
        n_train = n - 1
    indices = self._torch.randperm(n, device=self._device)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_dataset = TensorDataset(
        tokens[train_idx],
        charge_tensor[train_idx],
        ce_tensor[train_idx],
        target_tensor[train_idx],
    )
    val_dataset = TensorDataset(
        tokens[val_idx],
        charge_tensor[val_idx],
        ce_tensor[val_idx],
        target_tensor[val_idx],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    self.model.train()
    optimizer = self._torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    scheduler = self._torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, min_lr=1e-6
    )
    checkpoint = InMemoryCheckpoint(patience=patience)

    history = {"epochs": [], "train_loss": [], "val_loss": []}
    for epoch in range(epochs):
        self.model.train()
        train_loss = 0.0
        for tokens_b, charge_b, ce_b, target_b in train_loader:
            optimizer.zero_grad()
            outputs = self.model(
                tokens_b,
                charge=charge_b,
                collision_energy=ce_b,
            )
            pred = outputs['intensity'] if 'intensity' in outputs else list(outputs.values())[0]
            loss = masked_spectral_distance(target_b, pred)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        self.model.eval()
        val_loss = 0.0
        with self._torch.no_grad():
            for tokens_b, charge_b, ce_b, target_b in val_loader:
                outputs = self.model(
                    tokens_b,
                    charge=charge_b,
                    collision_energy=ce_b,
                )
                pred = outputs['intensity'] if 'intensity' in outputs else list(outputs.values())[0]
                val_loss += masked_spectral_distance(target_b, pred).item()
        val_loss /= max(len(val_loader), 1)
        scheduler.step(val_loss)

        history["epochs"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        if verbose and epoch % 5 == 0:
            print(
                f"Epoch {epoch}: intensity train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f}"
            )

        if checkpoint.step(val_loss, self.model):
            if verbose:
                print(f"Early stopping intensity fine-tune at epoch {epoch}")
            break

    checkpoint.restore(self.model)
    self.model.eval()
    self._finetune_history = history
DeepPeptideIntensityPredictor.fine_tune_model = _DeepPeptideIntensityPredictor_fine_tune_model

def _DeepPeptideIntensityPredictor_fine_tune_psms(
    self,
    psm_collection: List,
    batch_size: int = 64,
    epochs: int = 50,
    learning_rate: float = 1e-4,
    patience: int = 5,
    verbose: bool = False,
) -> None:
    """Fine-tune the native intensity model from Sage PSM observed fragments."""
    rows = []
    for psm in psm_collection:
        sequence = psm.sequence_modified if not psm.decoy else psm.sequence_decoy_modified
        target = observed_fragments_to_intensity_target(
            sequence,
            psm.charge,
            psm.sage_feature.fragments,
        )
        if np.any(target > 0):
            rows.append({
                'sequence': sequence,
                'charge': int(psm.charge),
                'collision_energy': float(psm.collision_energy),
                'intensity_target': target,
            })

    if len(rows) < 2:
        if verbose:
            print("Skipping intensity fine-tune: no usable fragment targets")
        return

    self.fine_tune_model(
        pd.DataFrame(rows),
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        patience=patience,
        verbose=verbose,
    )
DeepPeptideIntensityPredictor.fine_tune_psms = _DeepPeptideIntensityPredictor_fine_tune_psms

def _DeepPeptideIntensityPredictor_fine_tune_model_v2(
    self,
    data: pd.DataFrame,
    batch_size: int = 64,
    epochs: int = 50,
    learning_rate: float = 1e-4,
    patience: int = 5,
    divide_collision_energy_by: float = 1e2,
    verbose: bool = False,
) -> None:
    """Fine-tune the native intensity model on observed 174-vec targets.

    Required ``data`` columns:
        ``sequence`` (str, UNIMOD-bracket modified),
        ``charge`` (int),
        ``collision_energy`` (float),
        ``intensity_target`` (np.ndarray shape (174,) — observed
        intensities in the canonical ordinal-major layout; impossible
        ions marked -1; unobserved valid ions = 0).
    """
    assert 'sequence' in data.columns, 'Data must contain column "sequence"'
    assert 'charge' in data.columns, 'Data must contain column "charge"'
    assert 'collision_energy' in data.columns, 'Data must contain column "collision_energy"'
    assert 'intensity_target' in data.columns, 'Data must contain column "intensity_target"'

    from torch.utils.data import DataLoader, TensorDataset
    from pepdl_train.losses import masked_spectral_distance

    if len(data) < 2:
        if verbose:
            print("Skipping intensity fine-tune: need at least two PSMs")
        return

    sequences = data.sequence.tolist()
    charges = data.charge.astype(np.int64).tolist()
    collision_energies = (
        data.collision_energy.astype(float) / divide_collision_energy_by
    ).tolist()
    targets = np.vstack(data.intensity_target.to_numpy()).astype(np.float32)
    if targets.shape != (len(data), 174):
        raise ValueError(
            f"intensity_target must have shape (n, 174), got {targets.shape}"
        )

    tokens, charge_tensor, ce_tensor = self._preprocess(
        sequences, charges, collision_energies,
    )
    tokens = tokens.to(self._device)
    charge_tensor = charge_tensor.to(self._device)
    ce_tensor = ce_tensor.to(self._device)
    target_tensor = self._torch.tensor(
        targets, dtype=self._torch.float32, device=self._device,
    )

    n = len(sequences)
    # Group-aware (peptide × charge) split: same (modseq, charge) → same
    # fold. PSM-level random split would leak — the predictor is
    # deterministic per (sequence, charge, CE) so identical inputs in
    # train and val collapse val loss to the instrument's intensity-
    # noise floor, not the model's generalization.
    group_keys = np.array([f"{s}_{int(c)}" for s, c in zip(sequences, charges)])
    uniq, inv = np.unique(group_keys, return_inverse=True)
    n_groups = len(uniq)
    n_val_groups = max(1, int(n_groups * 0.2))
    if n_val_groups >= n_groups:
        n_val_groups = n_groups - 1
    rng_np = np.random.default_rng(42)
    perm_groups = rng_np.permutation(n_groups)
    val_groups = set(perm_groups[:n_val_groups].tolist())
    mask_val = np.fromiter((g in val_groups for g in inv),
                               dtype=bool, count=n)
    val_idx   = self._torch.from_numpy(np.flatnonzero(mask_val)).to(self._device)
    train_idx = self._torch.from_numpy(np.flatnonzero(~mask_val)).to(self._device)
    if verbose:
        print(f"[intens-ft] {n} PSMs ({n_groups:,} unique (modseq,charge)) → "
              f"train {len(train_idx):,}, val {len(val_idx):,} "
              f"(val groups: {n_val_groups:,})")

    train_ds = TensorDataset(
        tokens[train_idx], charge_tensor[train_idx],
        ce_tensor[train_idx], target_tensor[train_idx],
    )
    val_ds = TensorDataset(
        tokens[val_idx], charge_tensor[val_idx],
        ce_tensor[val_idx], target_tensor[val_idx],
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    self.model.train()
    optimizer = self._torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    scheduler = self._torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, min_lr=1e-6,
    )
    checkpoint = InMemoryCheckpoint(patience=patience)

    history = {"epochs": [], "train_loss": [], "val_loss": []}
    for epoch in range(epochs):
        self.model.train()
        train_loss = 0.0
        for tokens_b, charge_b, ce_b, target_b in train_loader:
            optimizer.zero_grad()
            outputs = self.model(
                tokens_b, charge=charge_b, collision_energy=ce_b,
            )
            pred = (outputs['intensity'] if 'intensity' in outputs
                      else list(outputs.values())[0])
            loss = masked_spectral_distance(target_b, pred)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        self.model.eval()
        val_loss = 0.0
        with self._torch.no_grad():
            for tokens_b, charge_b, ce_b, target_b in val_loader:
                outputs = self.model(
                    tokens_b, charge=charge_b, collision_energy=ce_b,
                )
                pred = (outputs['intensity'] if 'intensity' in outputs
                          else list(outputs.values())[0])
                val_loss += masked_spectral_distance(target_b, pred).item()
        val_loss /= max(len(val_loader), 1)
        scheduler.step(val_loss)

        history["epochs"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        if verbose and epoch % 5 == 0:
            print(
                f"Epoch {epoch}: intensity train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f}"
            )

        if checkpoint.step(val_loss, self.model):
            if verbose:
                print(f"Early stopping intensity fine-tune at epoch {epoch}")
            break

    checkpoint.restore(self.model)
    self.model.eval()
    self._finetune_history = history
DeepPeptideIntensityPredictor.fine_tune_model = _DeepPeptideIntensityPredictor_fine_tune_model_v2

def _DeepPeptideIntensityPredictor_fine_tune_psms_v2(
    self,
    psm_collection: List,
    batch_size: int = 64,
    epochs: int = 50,
    learning_rate: float = 1e-4,
    patience: int = 5,
    verbose: bool = False,
) -> None:
    """Fine-tune the intensity model on a list of sagepy PSM objects.

    Matches the signature ``sagepy-rescore`` calls
    (``fine_tune_psms(psms, batch_size=..., verbose=...)``). Labels are
    built from each PSM's matched fragments via
    :func:`observed_fragments_to_intensity_target`. CE is read from
    ``psm.collision_energy`` (rescore wrapper should have set the
    per-tile value beforehand).
    """
    rows = []
    for psm in psm_collection:
        sequence = (psm.sequence_modified if not psm.decoy
                      else psm.sequence_decoy_modified)
        target = observed_fragments_to_intensity_target(
            sequence,
            psm.charge,
            psm.sage_feature.fragments,
        )
        if np.any(target > 0):
            rows.append({
                'sequence': sequence,
                'charge': int(psm.charge),
                'collision_energy': float(psm.collision_energy),
                'intensity_target': target,
            })

    if len(rows) < 2:
        if verbose:
            print("Skipping intensity fine-tune: no usable fragment targets")
        return

    self.fine_tune_model(
        pd.DataFrame(rows),
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        patience=patience,
        verbose=verbose,
    )
DeepPeptideIntensityPredictor.fine_tune_psms = _DeepPeptideIntensityPredictor_fine_tune_psms_v2
