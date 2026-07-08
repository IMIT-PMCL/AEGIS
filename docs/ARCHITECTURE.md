# Architecture and scope of this release

This repository is the **public reference implementation** for the AEGIS manuscript.
It is deliberately scoped so that the work is fully **usable and reproducible**
without exposing the proprietary orchestration engine.

## What is included (open)

- **The evidence gate** (`aegis/gate.py`) — pre-registration, held-out generalization
  at a named unit, permutation null, threshold enforcement, mandatory negative
  reporting, and a checksummed provenance log. This is the core methodological
  contribution and is fully open.
- **Leakage-safe splitting** (`aegis/splits.py`) and **permutation nulls** (`aegis/nulls.py`).
- **The reference analyses** (`aegis/analyses/`) behind the figures: per-unit direction
  concordance, pan-cancer genotype-to-survival (log-rank + BH-FDR), leave-one-cancer-out
  universality, the optimal-cutpoint survival gate, and the rigor ablation.
- **A model-agnostic provider interface** (`aegis/llm/`) for the optional orchestration
  layer, plus a runnable command-line demo.

Every quantitative result in the paper is produced by these deterministic components.
**They do not require a language model** — the model is used only by the optional
orchestration layer to plan work; it never computes a scientific result.

## What is NOT included (proprietary)

- The internal **multi-agent coordinator** and **model-routing layer** of the
  production orchestration engine. These are an engineering system, not a source of
  scientific claims: they schedule and dispatch work but every claim is still computed
  by the deterministic analyses and the gate above. Because of that separation, the
  engine is **not needed to reproduce any result**.

Deployment of the full orchestration engine can be arranged with the corresponding
author under a research and evaluation agreement.

## Model-agnostic by design

No model vendor, product name, endpoint or credential appears anywhere in this
package. To use a model, implement `aegis.llm.LLMProvider`, or point the bundled
chat-completions-compatible adapter at any server via `AEGIS_LLM_BASE_URL` /
`AEGIS_LLM_API_KEY` / `AEGIS_LLM_MODEL`. Runs with no model at all in offline mode.

## Reproducing the public-data results

The DLBCL per-cell data are identifying and are governed by a data-use agreement;
the code that analyzes them is included, but the data are not. The public-data
analyses (pan-cancer survival, universality, the survival-cutpoint gate) can be run
directly on data retrieved from cBioPortal and UCSC Xena using the functions in
`aegis/analyses/`. Run `aegis demo` first to exercise the whole pipeline on synthetic
data with no downloads.
