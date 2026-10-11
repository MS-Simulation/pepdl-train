"""The fine-tune extensions pass the padding mask (and the predictor's instrument) on every call.

pepdl's predictors return ``(ids, padding_mask)`` / ``(tokens, padding_mask, charge, ce)`` since
pepdl#2, and the models must see that mask: inference without it put 2+ CCS +35 A^2 off and fragment
spectra at a median spectral angle of 0.43 instead of 0.83. Fine-tuning without it would train the
model on inputs it never sees at inference. Each test runs one epoch of a fine-tune on a tiny batch
with a stand-in model that records what every forward call received; no weights are needed.
"""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mscorepy")

import pepdl_train  # noqa: E402,F401  (attaches fine_tune_* to the predictors)

SEQS = ["PEPTIDEK", "K", "ACDEFGHIKLMNPQRSTVWYK", "LVNELTEFAK", "AGFAGDDAPR", "SAMPLERK"]


class _Recorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(1))
        self.calls = []
        self.max_seq_len = 50

    def _out(self, tokens, width):
        return self.w * torch.ones(tokens.shape[0], width)

    def predict_ccs(self, tokens, mz, charge, padding_mask=None, instrument=None):
        self.calls.append((tokens, padding_mask, instrument))
        return self._out(tokens, 1) * 400.0, torch.ones(tokens.shape[0], 1)

    def forward(self, tokens, charge=None, collision_energy=None, padding_mask=None,
                instrument=None, **kw):
        self.calls.append((tokens, padding_mask, instrument))
        if collision_energy is not None:
            return {"intensity": self._out(tokens, 174)}
        return {"rt": self._out(tokens, 1)}


def _predictor(module, name, instrument_id=None):
    import importlib

    from pepdl.utilities.tokenizers import ProformaTokenizer

    cls = getattr(importlib.import_module(module), name)
    obj = cls.__new__(cls)
    obj.tokenizer = ProformaTokenizer.with_defaults()
    obj._device = torch.device("cpu")
    obj._torch = torch
    obj.verbose = False
    obj.instrument_id = instrument_id
    obj.model = _Recorder()
    return obj


def _assert_masked(calls, want_instrument=None):
    from pepdl.utilities.padding import token_ids_and_mask
    from pepdl.utilities.tokenizers import ProformaTokenizer

    assert calls, "no forward call"
    tok = ProformaTokenizer.with_defaults()
    for tokens, mask, instrument in calls:
        assert mask is not None, "a forward call got no padding_mask"
        assert mask.dtype == torch.bool and mask.shape == tokens.shape
        # Each row's mask must be the mask of the row's own tokens: True exactly past its length.
        for row_ids, row_mask in zip(tokens.tolist(), mask.tolist()):
            n = row_mask.index(True) if True in row_mask else len(row_mask)
            assert all(row_mask[n:]) and not any(row_mask[:n])
            assert row_ids[n - 1] == tok.sep_token_id
        if want_instrument is None:
            assert instrument is None
        else:
            assert instrument is not None and set(instrument.tolist()) == {want_instrument}


def test_ccs_fine_tune_passes_the_mask():
    p = _predictor("pepdl.ccs.predictors", "DeepPeptideIonMobilityApex")
    data = pd.DataFrame({"sequence": SEQS, "sequence_modified": SEQS, "charge": [2, 2, 3, 2, 2, 3],
                         "calcmass": [900.0, 146.1, 2400.0, 1162.6, 975.5, 950.0],
                         "ims": [0.8, 0.7, 0.9, 0.85, 0.82, 0.75], "decoy": False})
    p.fine_tune_model(data, epochs=1, batch_size=4, decoys_separate=False)
    _assert_masked(p.model.calls)


def test_rt_fine_tune_passes_the_mask():
    p = _predictor("pepdl.rt.predictors", "DeepChromatographyApex")
    data = pd.DataFrame({"sequence": SEQS, "retention_time_projected": np.linspace(0, 1, len(SEQS)),
                         "decoy": False})
    p.fine_tune_model(data, epochs=1, batch_size=4, decoys_separate=False)
    _assert_masked(p.model.calls)


@pytest.mark.parametrize("instrument_id", [None, 1])
def test_intensity_fine_tune_passes_the_mask_and_the_instrument(instrument_id):
    p = _predictor("pepdl.intensity.predictors", "DeepPeptideIntensityPredictor", instrument_id)
    target = np.zeros(174, dtype=np.float32)
    target[:4] = [1.0, 0.5, 0.2, 0.1]
    data = pd.DataFrame({"sequence": SEQS, "charge": [2] * len(SEQS),
                         "collision_energy": [25.0] * len(SEQS),
                         "intensity_target": [target] * len(SEQS)})
    p.fine_tune_model(data, epochs=1, batch_size=4)
    _assert_masked(p.model.calls, want_instrument=instrument_id)
