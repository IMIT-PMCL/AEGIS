"""AEGIS: an evidence-gated reference implementation.

This package is the *public reference implementation* accompanying the AEGIS
manuscript. It contains the parts needed to understand, use and reproduce the
paper's evidence-gating methodology and its public-data analyses:

  * ``aegis.gate``        -- the evidence gate (pre-registration, held-out
                             generalization at a named unit, permutation null,
                             mandatory negative reporting, provenance).
  * ``aegis.splits``      -- grouped, leakage-safe held-out splitting.
  * ``aegis.nulls``       -- permutation-null utilities.
  * ``aegis.provenance``  -- append-only, checksummed decision log.
  * ``aegis.analyses``    -- reference analyses (per-cell direction concordance,
                             pan-cancer genotype-to-survival, leave-one-cancer-out
                             universality, optimal-cutpoint survival gate, and the
                             rigor ablation).
  * ``aegis.llm``         -- an OPTIONAL, model-agnostic provider interface for the
                             orchestration layer. The scientific analyses are fully
                             deterministic and require no language model.

The proprietary multi-agent orchestration engine (coordinator and model-routing
layer) is NOT part of this package. Every quantitative result is produced by the
deterministic analyses and the evidence gate that are included here, so the engine
is not required to reproduce any result. See ``docs/ARCHITECTURE.md``.
"""

from .gate import EvidenceGate, GateSpec, GateDecision
from .provenance import ProvenanceLog

__all__ = ["EvidenceGate", "GateSpec", "GateDecision", "ProvenanceLog"]
__version__ = "1.0.0"
