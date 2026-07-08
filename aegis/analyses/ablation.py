"""Rigor ablation on synthetic data.

Two self-contained demonstrations, on data with *no real genotype-phenotype signal*:

  * leakage: splitting by cell instead of by patient inflates a held-out AUC above
    chance; splitting by patient restores calibration.
  * calibration: a naive 'concordance > 0.5' rule false-flags null drivers, whereas
    gating on a permutation-null quantile holds the false-positive rate down.

Everything here is generated locally and needs no external data or model.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .concordance import make_concordance_statistic
from ..nulls import permutation_null, null_quantile


def make_synthetic_cohort(
    n_patients: int = 20,
    cells_per_patient: int = 200,
    n_genes: int = 100,
    offset_scale: float = 1.5,
    seed: int = 0,
):
    """Cells with a strong per-patient offset and a patient-level NULL label.

    The label is unrelated to expression biology, so any apparent predictive signal
    is leakage of patient identity.
    """
    rng = np.random.default_rng(seed)
    patient_offset = rng.normal(0, offset_scale, size=(n_patients, n_genes))
    patient_label = rng.integers(0, 2, size=n_patients)
    expr, mut, groups = [], [], []
    for p in range(n_patients):
        cells = patient_offset[p] + rng.normal(0, 1.0, size=(cells_per_patient, n_genes))
        expr.append(cells)
        mut.append(np.full(cells_per_patient, patient_label[p]))
        groups.append(np.full(cells_per_patient, p))
    return np.vstack(expr), np.concatenate(mut), np.concatenate(groups)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    r_pos = ranks[labels == 1].sum()
    auc = (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def _signature_auc(expr, mut, train_idx, test_idx, top_k=20) -> float:
    m = expr[train_idx][mut[train_idx] == 1]
    w = expr[train_idx][mut[train_idx] == 0]
    if len(m) == 0 or len(w) == 0:
        return 0.5
    delta = m.mean(axis=0) - w.mean(axis=0)
    top = np.argsort(-np.abs(delta))[:top_k]
    sig = delta[top] / (np.linalg.norm(delta[top]) + 1e-9)
    scores = expr[test_idx][:, top] @ sig
    return _auc(scores, mut[test_idx])


def leakage_demo(
    repeats: int = 15, n_patients: int = 30, cells_per: int = 100, n_genes: int = 100, seed: int = 0
) -> Dict[str, float]:
    """Held-out AUC on patient-level NULL labels, averaged over random splits.

    The label is a patient attribute unrelated to expression. Cell-level splitting
    memorizes patient identity (AUC well above chance); patient-level splitting
    cannot, so it sits near chance.
    """
    cell, patient = [], []
    for r in range(repeats):
        expr, mut, groups = make_synthetic_cohort(n_patients, cells_per, n_genes, seed=seed + r)
        rng = np.random.default_rng(1000 + seed + r)
        n = len(mut)

        perm = rng.permutation(n)
        half = n // 2
        cell.append(_signature_auc(expr, mut, perm[:half], perm[half:]))

        uniq = np.unique(groups)
        rng.shuffle(uniq)
        train_p = set(uniq[: len(uniq) // 2])
        tr = np.array([i for i in range(n) if groups[i] in train_p])
        te = np.array([i for i in range(n) if groups[i] not in train_p])
        patient.append(_signature_auc(expr, mut, tr, te))

    return {
        "auc_cell_level_split": float(np.mean(cell)),
        "auc_patient_level_split": float(np.mean(patient)),
    }


def calibration_demo(n_drivers: int = 20, n_perm: int = 100, seed: int = 0) -> Dict[str, float]:
    """False-positive rate on NULL drivers: naive 'concordance>0.5' vs permutation gate.

    Null drivers carry per-cell (within-patient) random genotype with no real effect,
    so a naive rule fires about half the time while the permutation gate holds near 5%.
    """
    expr, _, groups = make_synthetic_cohort(n_patients=15, cells_per_patient=80, n_genes=60, seed=seed)
    stat_fn = make_concordance_statistic(expr, top_k=20, min_per_class=5)
    rng = np.random.default_rng(seed + 1)
    n = len(groups)

    naive_fp = gated_fp = 0
    for d in range(n_drivers):
        labels = rng.integers(0, 2, size=n)  # per-cell null genotype (varies within patient)
        obs = stat_fn(labels, groups)
        null = permutation_null(lambda lab: stat_fn(lab, groups), labels,
                                n_perm=n_perm, groups=groups, seed=seed + d)
        q95 = null_quantile(null, 0.95)
        if obs > 0.5:
            naive_fp += 1
        if obs > q95:
            gated_fp += 1

    return {
        "naive_false_positive_rate": naive_fp / n_drivers,
        "gated_false_positive_rate": gated_fp / n_drivers,
    }
