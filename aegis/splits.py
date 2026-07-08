"""Leakage-safe, grouped held-out splitting.

The single most common failure in single-cell and multi-sample analyses is
splitting at the wrong unit (e.g. by individual cell) so that observations from
the same patient appear in both training and evaluation. These helpers enforce
splitting at a *named* unit (patient, donor, cancer type, ...).
"""

from __future__ import annotations

from typing import Iterator, List, Sequence, Tuple

import numpy as np


def unique_groups(groups: Sequence) -> List:
    seen = []
    s = set()
    for g in groups:
        if g not in s:
            s.add(g)
            seen.append(g)
    return seen


def leave_one_group_out(groups: Sequence) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` holding out one group at a time."""
    groups = np.asarray(groups, dtype=object)
    for g in unique_groups(list(groups)):
        test = np.where(groups == g)[0]
        train = np.where(groups != g)[0]
        if len(test) and len(train):
            yield train, test


def grouped_half_split(groups: Sequence, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Split groups (not rows) into two disjoint halves.

    Used by the optimal-cutpoint survival gate: model selection happens on one
    half of the *groups* and is evaluated on the other, so a selection that
    overfits noise is exposed on held-out data.
    """
    groups = np.asarray(groups, dtype=object)
    uniq = unique_groups(list(groups))
    rng = np.random.default_rng(seed)
    perm = list(uniq)
    rng.shuffle(perm)
    half = len(perm) // 2
    a = set(perm[:half])
    train = np.array([i for i, g in enumerate(groups) if g in a], dtype=int)
    test = np.array([i for i, g in enumerate(groups) if g not in a], dtype=int)
    return train, test
