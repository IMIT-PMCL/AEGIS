# AEGIS — evidence-gated reference implementation

Reference implementation accompanying **"AEGIS: an evidence-gated autonomous
co-scientist for trustworthy cancer genotype-to-phenotype discovery across secure
and public cohorts."**

AEGIS makes analytical rigor an **enforced** property rather than a matter of
discretion: a claim is reported only if, using a rule fixed **before** the result is
seen, it clears a pre-registered threshold **and** beats a permutation null, both on
**held-out** data at the biologically correct unit — with negatives reported by
default and every decision written to a checksummed provenance log.

This package contains the parts needed to **use and reproduce** that methodology. The
proprietary orchestration engine is not included and is not required to reproduce any
result — see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Install

```bash
pip install -e .            # core (numpy, scipy, pandas)
pip install -e ".[llm]"     # + optional model-agnostic orchestration adapter
pip install -e ".[dev]"     # + pytest
```

## Quickstart (offline, no data, no model)

```bash
aegis demo
```

This pre-registers a gate rule, **accepts** a driver with real patient-generalizing
signal, **reports a null driver as a negative**, and reproduces the leakage and
calibration ablations and the optimal-cutpoint survival gate — all on synthetic data,
writing `aegis_provenance.jsonl`.

## Use the gate on your own data

```python
import numpy as np
from aegis import EvidenceGate, GateSpec, ProvenanceLog
from aegis.analyses import make_concordance_statistic

# expr: (cells x genes) log-normalized ; mut: (cells,) 0/1 ; patient: (cells,) unit id
gate = EvidenceGate(ProvenanceLog("run.jsonl"), strict=True)
spec = GateSpec(
    name="MYD88 per-cell direction",
    unit="patient",                       # held-out unit (leakage-safe)
    metric="leave-one-patient-out sign-concordance",
    threshold=0.55,                       # the pre-registered bar
    null_quantile=0.95, n_perm=200, seed=0,
)
gate.preregister(spec)                     # <-- must precede the result

stat = make_concordance_statistic(expr, top_k=50, min_per_class=10)
decision = gate.evaluate(spec, mut, patient, stat)
print(decision.accepted, decision.observed, decision.null_quantile_value)
```

Calling `evaluate` before `preregister` raises — the gate cannot be bypassed.

## Public-data analyses

```python
from aegis.analyses import genotype_survival_scan, survival_cutpoint_gate
# genotype_survival_scan(df[cancer, driver, time, event, carrier]) -> log-rank + BH-FDR
# survival_cutpoint_gate(values, time, event) -> permutation-calibrated + held-out
```

Use these directly on data retrieved from cBioPortal (TCGA) and UCSC Xena (CPTAC-3).

## Bring your own model (optional)

The analyses need **no** language model. The optional orchestration layer is
model-agnostic — point it at any chat-completions-compatible server; no vendor or
product is referenced:

```bash
export AEGIS_LLM_BASE_URL="http://localhost:8000/v1"
export AEGIS_LLM_MODEL="your-model-identifier"
export AEGIS_LLM_API_KEY=""     # optional
```

or implement `aegis.llm.LLMProvider` for any backend.

## Tests

```bash
pytest -q
```

## License & citation

MIT (`LICENSE`). If you use this software, please cite the article (`CITATION.cff`).
