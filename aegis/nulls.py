"""Permutation-null utilities.

'Better than chance' is defined by the data, not by an assumed value. A statistic
is compared against a null distribution built by permuting the labels (optionally
within groups, to respect the dependence structure).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np


def permute_labels(
    labels: np.ndarray, groups: Optional[Sequence] = None, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Return a permuted copy of ``labels``.

    If ``groups`` is given, labels are permuted *within* each group so that the
    group structure (e.g. cells within a patient) is preserved under the null.
    """
    rng = rng or np.random.default_rng()
    labels = np.asarray(labels)
    if groups is None:
        return rng.permutation(labels)
    groups = np.asarray(groups, dtype=object)
    out = labels.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        out[idx] = rng.permutation(labels[idx])
    return out


def permutation_null(
    statistic_fn: Callable[[np.ndarray], float],
    labels: np.ndarray,
    n_perm: int = 200,
    groups: Optional[Sequence] = None,
    seed: int = 0,
) -> np.ndarray:
    """Build a null distribution of ``statistic_fn`` under permuted labels.

    ``statistic_fn`` must accept a label vector and return a scalar.
    """
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        null[i] = statistic_fn(permute_labels(labels, groups=groups, rng=rng))
    return null


def null_quantile(null: np.ndarray, q: float = 0.95) -> float:
    return float(np.quantile(np.asarray(null, dtype=float), q))


def empirical_p(observed: float, null: np.ndarray, greater_is_better: bool = True) -> float:
    null = np.asarray(null, dtype=float)
    n = len(null)
    if greater_is_better:
        hits = np.sum(null >= observed)
    else:
        hits = np.sum(null <= observed)
    return float((hits + 1) / (n + 1))
