"""Genotype-to-survival analyses.

Includes a dependency-light two-group log-rank test, a pan-cancer genotype-to-survival
scan with Benjamini-Hochberg control, and an optimal-cutpoint 'survival gate' that
demonstrates how held-out evaluation plus a permutation null tame the inflation from
selecting a cutpoint on the same data.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

try:  # scipy is a dependency; guard only to give a clear message if missing
    from scipy.stats import chi2 as _chi2
except Exception:  # pragma: no cover
    _chi2 = None


def logrank(time: Sequence, event: Sequence, group: Sequence) -> Dict[str, float]:
    """Two-group (Mantel-Cox) log-rank test. ``group`` is binary (0/1)."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    group = np.asarray(group, dtype=int)

    event_times = np.unique(time[event == 1])
    O1 = E1 = V = 0.0
    for t in event_times:
        at_risk = time >= t
        n = int(at_risk.sum())
        if n <= 1:
            continue
        n1 = int((at_risk & (group == 1)).sum())
        d = int(((time == t) & (event == 1)).sum())
        d1 = int(((time == t) & (event == 1) & (group == 1)).sum())
        E1 += d * n1 / n
        O1 += d1
        V += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)

    if V <= 0:
        return {"chi2": 0.0, "p": 1.0, "hr": float("nan"), "n1": int((group == 1).sum()),
                "n0": int((group == 0).sum()), "events": int(event.sum())}

    stat = (O1 - E1) ** 2 / V
    p = float(_chi2.sf(stat, 1)) if _chi2 is not None else float("nan")
    D = float(event.sum())
    O2, E2 = D - O1, D - E1
    hr = float((O1 / E1) / (O2 / E2)) if E1 > 0 and E2 > 0 and O2 > 0 else float("nan")
    return {"chi2": float(stat), "p": p, "hr": hr, "n1": int((group == 1).sum()),
            "n0": int((group == 0).sum()), "events": int(event.sum())}


def bh_fdr(pvals: Sequence) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(ranked, 0, 1)
    return out


def genotype_survival_scan(df, min_per_group: int = 5):
    """Per-(cancer, driver) log-rank scan with BH control.

    ``df`` must have columns: ``cancer``, ``driver``, ``time``, ``event`` (1/0),
    ``carrier`` (1/0). Returns a list of dicts with the statistic, hazard ratio and
    BH-adjusted p-value for every pair meeting the minimum group sizes.
    """
    import pandas as pd

    rows = []
    for (cancer, driver), sub in df.groupby(["cancer", "driver"]):
        g = sub["carrier"].astype(int).to_numpy()
        if (g == 1).sum() < min_per_group or (g == 0).sum() < min_per_group:
            continue
        res = logrank(sub["time"], sub["event"], g)
        rows.append({"cancer": cancer, "driver": driver, **res})
    if not rows:
        return pd.DataFrame(columns=["cancer", "driver", "chi2", "p", "hr", "q"])
    out = pd.DataFrame(rows)
    out["q"] = bh_fdr(out["p"].to_numpy())
    return out.sort_values("q").reset_index(drop=True)


def _max_chi2_over_cutpoints(values, time, event, cuts):
    best_chi2, best_cut = 0.0, None
    for c in cuts:
        grp = (np.asarray(values) > c).astype(int)
        if grp.sum() < 3 or (grp == 0).sum() < 3:
            continue
        stat = logrank(time, event, grp)["chi2"]
        if stat > best_chi2:
            best_chi2, best_cut = stat, c
    return best_chi2, best_cut


def survival_cutpoint_gate(
    values: Sequence,
    time: Sequence,
    event: Sequence,
    n_cuts: int = 7,
    n_perm: int = 300,
    seed: int = 0,
    train_frac: float = 0.5,
) -> Dict[str, float]:
    """Optimal-cutpoint survival analysis, gated.

    Searching over cutpoints inflates significance; this reports both the naive
    optimized statistic and (i) a permutation-null-calibrated threshold and (ii) a
    held-out p-value (cutpoint chosen on a training split, evaluated on the rest).
    """
    values = np.asarray(values, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    rng = np.random.default_rng(seed)
    cuts = np.quantile(values, np.linspace(0.2, 0.8, n_cuts))

    obs_chi2, _ = _max_chi2_over_cutpoints(values, time, event, cuts)

    null = np.empty(n_perm)
    idx = np.arange(len(values))
    for i in range(n_perm):
        perm = rng.permutation(idx)  # shuffle survival relative to values
        null[i], _ = _max_chi2_over_cutpoints(values, time[perm], event[perm], cuts)
    null_q95 = float(np.quantile(null, 0.95))

    n = len(values)
    tr = rng.permutation(n)
    k = int(round(train_frac * n))
    tr_idx, te_idx = tr[:k], tr[k:]
    _, best_cut = _max_chi2_over_cutpoints(values[tr_idx], time[tr_idx], event[tr_idx], cuts)
    if best_cut is None:
        holdout_p = 1.0
    else:
        grp_te = (values[te_idx] > best_cut).astype(int)
        holdout_p = logrank(time[te_idx], event[te_idx], grp_te)["p"]

    return {
        "observed_chi2": float(obs_chi2),
        "null_q95_chi2": null_q95,
        "beats_null": bool(obs_chi2 > null_q95),
        "holdout_p": float(holdout_p),
    }
