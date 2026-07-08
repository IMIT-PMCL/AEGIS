"""Per-unit direction concordance (leave-one-group-out).

For a binary genotype label and an expression matrix, we ask whether the direction
of the mutated-versus-wild-type expression shift learned on training units agrees
with the shift measured on a held-out unit. The score is the mean, over held-out
units, of the sign-concordance of the top training-effect genes.

Setting the group to *patient* reproduces the per-cell analysis; setting it to
*cancer type* reproduces the leave-one-cancer-out universality analysis.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from ..splits import leave_one_group_out


def _delta(expr: np.ndarray, mut: np.ndarray) -> np.ndarray:
    """Per-gene mean difference (mutated minus wild-type)."""
    m = expr[mut == 1]
    w = expr[mut == 0]
    if len(m) == 0 or len(w) == 0:
        return np.zeros(expr.shape[1])
    return m.mean(axis=0) - w.mean(axis=0)


def leave_one_group_out_sign_concordance(
    expr: np.ndarray,
    mut: np.ndarray,
    groups: Sequence,
    top_k: int = 50,
    min_per_class: int = 10,
) -> float:
    """Mean leave-one-group-out sign-concordance.

    Parameters
    ----------
    expr : (n, g) array of (log-normalized) expression.
    mut  : (n,) binary genotype labels.
    groups : (n,) unit labels (e.g. patient or cancer type).
    top_k : number of top training-effect genes scored per fold.
    min_per_class : minimum mutated and wild-type observations required in the
        held-out unit for the fold to count.
    """
    expr = np.asarray(expr, dtype=float)
    mut = np.asarray(mut).astype(int)
    scores = []
    for train, test in leave_one_group_out(groups):
        y_test = mut[test]
        if (y_test == 1).sum() < min_per_class or (y_test == 0).sum() < min_per_class:
            continue
        d_train = _delta(expr[train], mut[train])
        k = min(top_k, expr.shape[1])
        top = np.argsort(-np.abs(d_train))[:k]
        d_test = _delta(expr[test], mut[test])
        s_train = np.sign(d_train[top])
        s_test = np.sign(d_test[top])
        valid = s_train != 0
        if valid.sum() == 0:
            continue
        scores.append(float(np.mean(s_train[valid] == s_test[valid])))
    return float(np.mean(scores)) if scores else 0.5


def make_concordance_statistic(
    expr: np.ndarray, top_k: int = 50, min_per_class: int = 10
) -> Callable[[np.ndarray, Sequence], float]:
    """Return a ``statistic_fn(labels, groups)`` for :meth:`EvidenceGate.evaluate`."""

    expr = np.asarray(expr, dtype=float)

    def _fn(labels: np.ndarray, groups: Sequence) -> float:
        return leave_one_group_out_sign_concordance(
            expr, labels, groups, top_k=top_k, min_per_class=min_per_class
        )

    return _fn
