"""Sanity tests for the evidence gate and analyses (run with: pytest)."""

import numpy as np

from aegis.gate import EvidenceGate, GateSpec, GateError
from aegis.analyses.concordance import make_concordance_statistic
from aegis.analyses.ablation import leakage_demo, calibration_demo
from aegis.analyses.survival import logrank


def _cohort(effect, seed=0, n_patients=16, cells=150, genes=80):
    rng = np.random.default_rng(seed)
    offset = rng.normal(0, 1.5, (n_patients, genes))
    direction = np.zeros(genes); direction[:15] = rng.choice([-1.0, 1.0], 15)
    expr, mut, groups = [], [], []
    for p in range(n_patients):
        c = offset[p] + rng.normal(0, 1, (cells, genes))
        y = rng.integers(0, 2, cells)
        c[y == 1] += effect * direction
        expr.append(c); mut.append(y); groups.append(np.full(cells, p))
    return np.vstack(expr), np.concatenate(mut), np.concatenate(groups)


def test_gate_requires_preregistration():
    expr, mut, groups = _cohort(1.0)
    gate = EvidenceGate(strict=True)
    spec = GateSpec(name="x", unit="patient", metric="concordance", threshold=0.55, n_perm=30)
    stat = make_concordance_statistic(expr, top_k=15, min_per_class=5)
    try:
        gate.evaluate(spec, mut, groups, stat)
        assert False, "should have refused an unregistered spec"
    except GateError:
        pass


def test_real_signal_accepted_null_rejected():
    gate = EvidenceGate(strict=True)
    spec = GateSpec(name="x", unit="patient", metric="concordance", threshold=0.55, n_perm=50)
    gate.preregister(spec)

    expr, mut, groups = _cohort(1.5, seed=1)
    stat = make_concordance_statistic(expr, top_k=15, min_per_class=5)
    assert gate.evaluate(spec, mut, groups, stat).accepted

    rng = np.random.default_rng(2)
    uniq = np.unique(groups)
    null_lab = rng.integers(0, 2, len(uniq))[np.searchsorted(uniq, groups)]
    assert not gate.evaluate(spec, null_lab, groups, stat).accepted


def test_leakage_ablation_direction():
    r = leakage_demo(seed=0)
    # cell-level split leaks patient identity -> inflated above patient-level
    assert r["auc_cell_level_split"] > r["auc_patient_level_split"]


def test_calibration_ablation_direction():
    r = calibration_demo(n_drivers=15, n_perm=60, seed=0)
    assert r["gated_false_positive_rate"] <= r["naive_false_positive_rate"]


def test_logrank_runs():
    rng = np.random.default_rng(0)
    res = logrank(rng.exponential(size=100), rng.integers(0, 2, 100), rng.integers(0, 2, 100))
    assert 0.0 <= res["p"] <= 1.0
