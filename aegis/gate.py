"""The evidence gate.

A claim is *accepted* only if, using a rule fixed **before** the result is seen,
it (i) meets a pre-registered threshold and (ii) exceeds a permutation null, both
evaluated on held-out data at the biologically correct unit. Failures are recorded
as negatives rather than silently dropped or re-run until they pass. The gate is a
hard constraint the caller cannot bypass -- not advice a model can choose to ignore.

The ``statistic_fn`` you pass is expected to already perform held-out evaluation at
the correct unit (see ``aegis.splits`` and the reference analyses). The gate adds
pre-registration, the permutation null, threshold enforcement and provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Optional, Sequence

import numpy as np

from .nulls import permutation_null, null_quantile, empirical_p
from .provenance import ProvenanceLog, sha256_of


@dataclass(frozen=True)
class GateSpec:
    """A pre-registered decision rule. Frozen so it can be content-addressed."""

    name: str
    unit: str                      # the held-out unit, e.g. "patient" or "cancer_type"
    metric: str                    # human-readable metric name
    threshold: float               # the pre-registered bar
    greater_is_better: bool = True
    null_quantile: float = 0.95    # a claim must beat this quantile of the null
    n_perm: int = 200
    seed: int = 0
    notes: str = ""

    def hash(self) -> str:
        return sha256_of(asdict(self))


@dataclass
class GateDecision:
    spec_name: str
    spec_hash: str
    accepted: bool
    observed: float
    threshold: float
    null_quantile_value: float
    empirical_p: float
    passed_threshold: bool
    beat_null: bool
    reason: str


class GateError(RuntimeError):
    pass


class EvidenceGate:
    def __init__(self, provenance: Optional[ProvenanceLog] = None, strict: bool = True):
        self.provenance = provenance
        self.strict = strict
        self._registered = set()

    # -- pre-registration must happen before any result is generated -----------
    def preregister(self, spec: GateSpec) -> str:
        h = spec.hash()
        self._registered.add(h)
        if self.provenance:
            self.provenance.append("preregistration", asdict(spec) | {"spec_hash": h})
        return h

    # -- evaluation ------------------------------------------------------------
    def evaluate(
        self,
        spec: GateSpec,
        labels: Sequence,
        groups: Sequence,
        statistic_fn: Callable[[np.ndarray, Sequence], float],
    ) -> GateDecision:
        """Evaluate one claim through the gate and record the decision.

        ``statistic_fn(labels, groups) -> float`` must compute the held-out
        statistic at the named unit. It is called once for the observed value and
        ``n_perm`` times on permuted labels to build the null.
        """
        h = spec.hash()
        if self.strict and h not in self._registered:
            raise GateError(
                f"Spec '{spec.name}' was not pre-registered before evaluation. "
                "Call preregister(spec) first (or set strict=False for exploration)."
            )

        labels = np.asarray(labels)
        observed = float(statistic_fn(labels, groups))
        null = permutation_null(
            lambda lab: float(statistic_fn(lab, groups)),
            labels,
            n_perm=spec.n_perm,
            groups=groups,
            seed=spec.seed,
        )
        nq = null_quantile(null, spec.null_quantile)
        p = empirical_p(observed, null, greater_is_better=spec.greater_is_better)

        if spec.greater_is_better:
            passed_threshold = observed >= spec.threshold
            beat_null = observed > nq
        else:
            passed_threshold = observed <= spec.threshold
            beat_null = observed < nq

        accepted = bool(passed_threshold and beat_null)
        reason = (
            "accepted: cleared threshold and permutation null on held-out data"
            if accepted
            else "reported as negative: "
            + ("did not clear threshold; " if not passed_threshold else "")
            + ("did not beat permutation null" if not beat_null else "")
        ).strip("; ")

        decision = GateDecision(
            spec_name=spec.name,
            spec_hash=h,
            accepted=accepted,
            observed=observed,
            threshold=spec.threshold,
            null_quantile_value=nq,
            empirical_p=p,
            passed_threshold=passed_threshold,
            beat_null=beat_null,
            reason=reason,
        )
        if self.provenance:
            self.provenance.append("decision", asdict(decision))
        return decision
