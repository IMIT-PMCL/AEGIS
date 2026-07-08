"""Command-line entry point.

``aegis demo`` runs the whole methodology end-to-end on synthetic data -- no private
data, no network, no language model required -- and writes a provenance log:

  1. pre-registers an evidence-gate rule;
  2. accepts a driver with real, patient-generalizing signal;
  3. reports a null driver as a negative;
  4. reproduces the leakage and calibration ablations;
  5. runs the optimal-cutpoint survival gate.
"""

from __future__ import annotations

import argparse

import numpy as np

from . import __version__
from .gate import EvidenceGate, GateSpec
from .provenance import ProvenanceLog
from .analyses.concordance import make_concordance_statistic
from .analyses.ablation import leakage_demo, calibration_demo
from .analyses.survival import survival_cutpoint_gate


def _signal_cohort(n_patients=20, cells_per=200, n_genes=100, effect=1.0, seed=0):
    """Synthetic cohort with a REAL, patient-generalizing mutated-vs-WT direction."""
    rng = np.random.default_rng(seed)
    offset = rng.normal(0, 1.5, size=(n_patients, n_genes))
    direction = np.zeros(n_genes)
    direction[:20] = rng.choice([-1.0, 1.0], size=20)
    expr, mut, groups = [], [], []
    for p in range(n_patients):
        cells = offset[p] + rng.normal(0, 1.0, size=(cells_per, n_genes))
        y = rng.integers(0, 2, size=cells_per)          # per-cell genotype
        cells[y == 1] += effect * direction              # real shift in mutated cells
        expr.append(cells); mut.append(y); groups.append(np.full(cells_per, p))
    return np.vstack(expr), np.concatenate(mut), np.concatenate(groups)


def _null_labels(groups, seed=1):
    # per-cell null genotype: varies within patient, so held-out folds are evaluable
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=len(groups))


def cmd_demo(args) -> None:
    prov = ProvenanceLog(args.provenance)
    gate = EvidenceGate(prov, strict=True)

    expr, mut, groups = _signal_cohort(seed=0)
    stat = make_concordance_statistic(expr, top_k=20, min_per_class=5)

    spec = GateSpec(name="per-cell mutated-vs-WT direction", unit="patient",
                    metric="leave-one-patient-out sign-concordance",
                    threshold=0.55, n_perm=args.n_perm, seed=0)
    gate.preregister(spec)  # rule fixed BEFORE any result

    print(f"AEGIS reference demo (v{__version__})")
    print("-" * 64)
    print(f"Pre-registered rule: {spec.metric} > {spec.threshold} at unit '{spec.unit}',")
    print(f"                     must beat the {int(spec.null_quantile*100)}th pct of a permutation null.\n")

    d_real = gate.evaluate(spec, mut, groups, stat)
    print(f"[real driver]  concordance={d_real.observed:.3f}  null95={d_real.null_quantile_value:.3f}"
          f"  p={d_real.empirical_p:.3g}  -> {'ACCEPTED' if d_real.accepted else 'negative'}")

    d_null = gate.evaluate(spec, _null_labels(groups), groups, stat)
    print(f"[null driver]  concordance={d_null.observed:.3f}  null95={d_null.null_quantile_value:.3f}"
          f"  p={d_null.empirical_p:.3g}  -> {'ACCEPTED' if d_null.accepted else 'reported as negative'}")

    print("\nAblation (data with no real signal):")
    lk = leakage_demo(seed=0)
    print(f"  leakage    : held-out AUC  cell-level={lk['auc_cell_level_split']:.2f}"
          f"  vs patient-level={lk['auc_patient_level_split']:.2f}")
    cal = calibration_demo(n_drivers=args.n_drivers, n_perm=args.n_perm, seed=0)
    print(f"  calibration: false positives  naive(>0.5)={cal['naive_false_positive_rate']:.0%}"
          f"  vs gated={cal['gated_false_positive_rate']:.0%}")

    print("\nOptimal-cutpoint survival gate (synthetic, no real effect):")
    rng = np.random.default_rng(7)
    n = 400
    sg = survival_cutpoint_gate(rng.normal(size=n),
                                rng.exponential(size=n),
                                rng.integers(0, 2, size=n),
                                n_perm=args.n_perm, seed=7)
    print(f"  observed chi2={sg['observed_chi2']:.2f}  null95={sg['null_q95_chi2']:.2f}"
          f"  beats_null={sg['beats_null']}  held-out p={sg['holdout_p']:.2g}")

    print(f"\nProvenance written to: {args.provenance}")


def cmd_version(args) -> None:
    print(__version__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aegis", description="AEGIS evidence-gated reference implementation")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="run the end-to-end methodology on synthetic data")
    d.add_argument("--provenance", default="aegis_provenance.jsonl")
    d.add_argument("--n-perm", type=int, default=100)
    d.add_argument("--n-drivers", type=int, default=20)
    d.set_defaults(func=cmd_demo)

    v = sub.add_parser("version", help="print version")
    v.set_defaults(func=cmd_version)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
