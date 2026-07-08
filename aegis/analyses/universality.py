"""Leave-one-cancer-out universality.

Whether a driver's transcriptional effect is shared across cancers is tested by
predicting each held-out cancer's effect direction from the mean of the other
cancers and scoring sign-concordance -- i.e. the same concordance machinery with
the held-out unit set to *cancer type*.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .concordance import make_concordance_statistic


def make_universality_statistic(
    expr: np.ndarray, top_k: int = 100, min_per_class: int = 8
) -> Callable[[np.ndarray, Sequence], float]:
    """Return a ``statistic_fn(labels, cancer_types)`` for the evidence gate.

    Identical to the per-cell concordance statistic but intended to be called with
    cancer-type groups and a wider gene panel, matching the manuscript's
    leave-one-cancer-out test.
    """
    return make_concordance_statistic(expr, top_k=top_k, min_per_class=min_per_class)
