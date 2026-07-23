"""pepdl-train.rescoring — sagepy PSM-facing predict-with-refinement functions extracted from pepdl
inference (they fine-tune on Sage PSMs, so they are rescoring/training, not inference). They rely on the
monkeypatched fine_tune_* (import pepdl_train first) and on sagepy via pepdl.lazy_imports.
Provenance: verbatim from imspy-predictors ccs/rt/intensity predictors."""
from __future__ import annotations
import numpy as np
import pandas as pd

import pepdl_train  # noqa: F401  (ensures fine_tune_* is attached before these call it)
from pepdl.lazy_imports import get_sagepy_psm_utils, get_sagepy_fragment_utils
from pepdl.ccs.predictors import DeepPeptideIonMobilityApex, load_deep_ccs_predictor
from pepdl.rt.predictors import DeepChromatographyApex, load_deep_retention_time_predictor
from pepdl.intensity.predictors import DeepPeptideIntensityPredictor

# --- from pepdl ccs/predictors.py ---
def predict_inverse_ion_mobility(
    psm_collection: List,
    refine_model: bool = True,
    verbose: bool = False,
) -> None:
    """
    Predict inverse ion mobility for a collection of peptide spectrum matches.

    Note: This function requires sagepy (via imspy-search package).

    Args:
        psm_collection: A list of peptide spectrum matches (sagepy Psm objects).
        refine_model: Whether to refine the model by fine-tuning it on the provided data.
        verbose: Whether to print additional information during the prediction.

    Returns:
        None, the inverse ion mobility is set in the peptide spectrum matches in place.
    """
    Psm, psm_collection_to_pandas = get_sagepy_psm_utils()
    generate_balanced_im_dataset = get_search_im_utils()

    im_predictor = DeepPeptideIonMobilityApex(verbose=verbose)

    if refine_model:
        im_predictor.fine_tune_model(
            psm_collection_to_pandas(generate_balanced_im_dataset(psm_collection)),
            batch_size=128,
            verbose=verbose,
        )

    # predict ion mobilities
    inv_mob = im_predictor.simulate_ion_mobilities(
        sequences=[
            x.sequence_modified if not x.decoy else x.sequence_decoy_modified
            for x in psm_collection
        ],
        charges=[x.charge for x in psm_collection],
        mz=[x.mono_mz_calculated for x in psm_collection],
    )

    # set ion mobilities
    for mob, ps in zip(inv_mob, psm_collection):
        ps.inverse_ion_mobility_predicted = mob

# --- from pepdl rt/predictors.py ---
def predict_retention_time(
    psm_collection: List,
    refine_model: bool = True,
    verbose: bool = False,
) -> None:
    """
    Predict retention times for a collection of peptide-spectrum matches.

    Note: This function requires sagepy (via imspy-search package).

    Args:
        psm_collection: A list of peptide-spectrum matches (sagepy Psm objects)
        refine_model: Whether to refine the model
        verbose: Whether to print verbose output

    Returns:
        None, retention times are set in the peptide-spectrum matches
    """
    Psm, psm_collection_to_pandas = get_sagepy_psm_utils()
    generate_balanced_rt_dataset = get_search_rt_utils()

    rt_predictor = DeepChromatographyApex(verbose=verbose)

    rt_min = np.min([x.retention_time for x in psm_collection])
    rt_max = np.max([x.retention_time for x in psm_collection])

    for psm in psm_collection:
        psm.retention_time_projected = linear_map(
            psm.retention_time, rt_min, rt_max, 0, 60
        )

    if refine_model:
        rt_predictor.fine_tune_model(
            psm_collection_to_pandas(generate_balanced_rt_dataset(psm_collection)),
            batch_size=128,
            verbose=verbose,
        )

    # Predict retention times
    rt_predicted = rt_predictor.simulate_separation_times(
        sequences=[
            x.sequence_modified if not x.decoy else x.sequence_decoy_modified
            for x in psm_collection
        ],
    )

    # Set the predicted retention times
    for rt, ps in zip(rt_predicted, psm_collection):
        ps.retention_time_predicted = rt

# --- from pepdl intensity/predictors.py ---
def predict_intensities_prosit(
        psm_collection: List,
        calibrate_collision_energy: bool = True,
        verbose: bool = False,
        num_threads: int = -1,
) -> None:
    """
    Predict the fragment ion intensities using Prosit via Koina.

    Note: This function requires sagepy (via imspy-search package).

    Args:
        psm_collection: a list of peptide-spectrum matches (sagepy Psm objects)
        calibrate_collision_energy: whether to calibrate the collision energy
        verbose: whether to print progress
        num_threads: number of threads to use

    Returns:
        None, the fragment ion intensities are stored in the PeptideSpectrumMatch objects
    """
    associate_fragment_ions_with_prosit_predicted_intensities, Psm = get_sagepy_fragment_utils()

    # check if num_threads is -1, if so, use all available threads
    if num_threads == -1:
        num_threads = os.cpu_count()

    # the intensity predictor model
    prosit_model = Prosit2023TimsTofWrapper(verbose=False)

    # Calibrate one absolute NCE for the run (calibrate_nce drops decoys and
    # caps the sample internally). The model conditions on a per-run NCE, so
    # every PSM is predicted at that single value -- not observed CE + offset.
    if calibrate_collision_energy:
        calibration = calibrate_nce(prosit_model, psm_collection, verbose=verbose)
        calibrated_nce = float(calibration["best_nce"])
        for ps in psm_collection:
            ps.collision_energy_calibrated = calibrated_nce
    else:
        for ps in psm_collection:
            ps.collision_energy_calibrated = ps.collision_energy

    intensity_pred = prosit_model.predict_intensities(
        [p.sequence_modified for p in psm_collection],
        np.array([p.charge for p in psm_collection]),
        [p.collision_energy_calibrated for p in psm_collection],
        batch_size=2048,
        flatten=True,
    )

    psm_collection_intensity = associate_fragment_ions_with_prosit_predicted_intensities(
        psm_collection, intensity_pred, num_threads=num_threads
    )

    # calculate the spectral similarity metrics
    for psm, psm_intensity in tqdm(zip(psm_collection, psm_collection_intensity),
                                                      desc='Calc spectral similarity metrics', ncols=100, disable=not verbose):
        psm.prosit_predicted_intensities = psm_intensity.prosit_predicted_intensities

# --- from pepdl intensity/predictors.py ---
def calibrate_nce(
        model,
        psms: List,
        nce_grid: Optional[List[int]] = None,
        per_charge: bool = False,
        max_sample: int = 2048,
        verbose: bool = False,
) -> dict:
    """Calibrate the absolute normalized collision energy (NCE) for a run.

    The intensity model conditions on a per-run NCE scalar -- it was fine-tuned
    on ``collision_energy_aligned_normed`` (domain ~7-43). Calibration sweeps
    absolute NCE values, predicts every PSM at each, and returns the value that
    maximizes the mean predicted-vs-observed spectral angle.

    This is an *absolute* sweep, NOT an offset on the observed collision energy:
    the observed CE (e.g. the Bruker mobility-ramped value) is a different
    physical quantity and must not be added to. One NCE is returned per run.

    Note: This function requires sagepy (via the imspy-search package).

    Args:
        model: an intensity predictor exposing ``predict_intensities(sequences,
            charges, collision_energies, batch_size=, flatten=True)`` -- e.g.
            Prosit2023TimsTofWrapper or DeepPeptideIntensityPredictor.
        psms: sagepy Psm objects carrying observed fragments. Decoys are dropped.
        nce_grid: absolute NCE values to sweep (default ``range(15, 51)``).
        per_charge: also report the best NCE separately per precursor charge.
        max_sample: cap the calibration sample; if exceeded, the highest-scoring
            PSMs (by hyperscore) are kept.
        verbose: whether to print progress.

    Returns:
        dict: ``{best_nce, curve: [(nce, mean_spectral_angle), ...], n_psms}``;
        also ``per_charge: {charge: best_nce}`` when ``per_charge`` is True.
    """
    associate_fragment_ions_with_prosit_predicted_intensities, _ = get_sagepy_fragment_utils()

    if nce_grid is None:
        nce_grid = list(range(15, 51))
    nce_grid = [int(x) for x in nce_grid]

    targets = [p for p in psms if not getattr(p, "decoy", False)]
    if not targets:
        raise ValueError("calibrate_nce: no target PSMs to calibrate on")
    if max_sample and len(targets) > max_sample:
        targets = sorted(targets, key=lambda p: getattr(p, "hyperscore", 0.0),
                         reverse=True)[:max_sample]

    def _sweep(sample):
        # sequence_modified (not sequence) -- the prediction is mod-aware.
        seqs = [p.sequence_modified for p in sample]
        chgs = np.array([p.charge for p in sample])
        curve = []
        for nce in tqdm(nce_grid, disable=not verbose, desc="calibrating NCE", ncols=100):
            intensities = model.predict_intensities(
                seqs, chgs, [float(nce)] * len(sample),
                batch_size=2048, flatten=True,
            )
            scored = associate_fragment_ions_with_prosit_predicted_intensities(
                sample, intensities
            )
            sa = float(np.mean([x.spectral_angle_similarity for x in scored]))
            curve.append((int(nce), sa))
        best = curve[int(np.argmax([s for _, s in curve]))][0]
        return int(best), curve

    best_nce, curve = _sweep(targets)
    result = {"best_nce": best_nce, "curve": curve, "n_psms": len(targets)}

    if per_charge:
        pc = {}
        for z in sorted({int(p.charge) for p in targets}):
            sub = [p for p in targets if int(p.charge) == z]
            if len(sub) < 100:
                continue
            pc[z], _ = _sweep(sub)
        result["per_charge"] = pc

    if verbose:
        print(f"calibrate_nce: best NCE = {best_nce} "
              f"(mean spectral angle {max(s for _, s in curve):.4f}, "
              f"n = {len(targets)})")
    return result
