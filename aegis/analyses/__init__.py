"""Reference analyses used in the manuscript.

These implement the statistics behind the paper's figures. They operate on generic
matrices/tables so they can be run on your own inputs or on public data. Each is
written to be composed with :class:`aegis.gate.EvidenceGate`.
"""

from .concordance import (
    leave_one_group_out_sign_concordance,
    make_concordance_statistic,
)
from .survival import logrank, genotype_survival_scan, survival_cutpoint_gate, bh_fdr
from .universality import make_universality_statistic

__all__ = [
    "leave_one_group_out_sign_concordance",
    "make_concordance_statistic",
    "logrank",
    "genotype_survival_scan",
    "survival_cutpoint_gate",
    "bh_fdr",
    "make_universality_statistic",
]
