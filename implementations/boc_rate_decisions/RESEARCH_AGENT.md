# Research-grounded BoC agent

This document explains the cutoff-aware research layer used by the Bank of
Canada rate-decision agent. It covers the implementation in
`research.py`, its integration with the forecast context and prompt builder,
and the recommended testing workflow.

## Purpose

The research-grounded predictor forecasts whether the Bank of Canada will
cut, hold, or hike its target rate at the next fixed announcement date. It
combines the existing quantitative payload with cached Bank communications
that were public at the forecast origin.

The implementation has three design goals:

1. **No document leakage.** A forecast cannot see a document published after
   its `as_of` date.
2. **Reproducibility.** Historical runs use cached documents rather than live
   web-search results.
3. **Bounded context.** The number and size of documents included in a prompt
   are configurable.

## Architecture

```text
Cached press-release artifacts
          │
          ▼
DocumentStore
          │ cutoff: publication_date <= as_of
          ▼
ForecastContext.get_documents()
          │
          ▼
research.py
  ├── merge configured sources
  ├── select newest documents
  ├── truncate document text
  └── preserve provenance
          │
          ▼
BoCDecisionPromptBuilder
          │ quantitative data + research_evidence
          ▼
AgentPredictor
          │
          ▼
P(cut), P(hold), P(hike) + rationale + key signals
```

Cutoff enforcement happens in `ForecastContext`, not in the agent. The agent
therefore receives only documents already deemed visible at the forecast
origin.

## Files

| File | Responsibility |
|---|---|
| `research.py` | Select and format cutoff-visible documents for prompts. |
| `press_releases.py` | Acquire, extract, cache, and inspect BoC announcement releases. |
| `data.py` | Attach a `DocumentStore` to the BoC `DataService`. |
| `analyst_agent/agent.py` | Build quantitative and research-grounded prompts and predictors. |
| `analyst_agent/__init__.py` | Export the public research-agent factories. |
| `tests/boc_rate_decisions/test_research.py` | Test source merging, cutoff behavior, provenance, and prompt budgets. |

## Research evidence model

`format_research_evidence()` returns a list of JSON-serializable dictionaries:

```python
{
    "source": "boc_press_releases",
    "document_id": "2024-04-10_en",
    "title": "Bank of Canada rate announcement 2024-04-10",
    "publication_date": "2024-04-10",
    "text": "...",
}
```

The provenance fields let the agent cite stable document identifiers in its
`key_signals`. They also make forecast traces auditable after a decision
resolves.

## Public API

### `latest_research_documents()`

```python
latest_research_documents(
    context,
    sources=("boc_press_releases",),
    max_documents=3,
)
```

This function:

- asks the context for already cutoff-filtered documents;
- merges documents from all requested sources;
- sorts them by publication date, source, and document ID;
- returns the newest `max_documents` items.

A zero document budget returns an empty list. A negative budget raises
`ValueError`.

### `format_research_evidence()`

```python
format_research_evidence(
    context,
    sources=("boc_press_releases",),
    max_documents=3,
    max_chars_per_document=6_000,
)
```

This converts selected documents into prompt-ready dictionaries. Documents
larger than `max_chars_per_document` are truncated and marked with
`[truncated]`.

### `build_boc_research_predictor()`

```python
build_boc_research_predictor(
    model=LITE_MODEL,
    max_documents=3,
    max_chars_per_document=6_000,
)
```

This is the main predictor factory. It combines:

- `build_boc_research_config()`;
- a document-aware `BoCDecisionPromptBuilder`;
- `CategoricalAgentForecastOutput`;
- the standard `AgentPredictor` interface.

## Setup

Install the workspace and fetch both numeric and document data from the
repository root:

```bash
uv sync
uv run python scripts/fetch_boc.py
uv run python scripts/fetch_boc_press_releases.py
```

The numeric cache is stored under `data/statcan` and `data/fred`. Extracted
press releases are stored under `data/reports/boc_press_releases`. The entire
`data/` directory is gitignored.

The unemployment series requires `FRED_API_KEY` in the repository-root `.env`.

## Constructing the service and predictor

```python
from pathlib import Path

from boc_rate_decisions.analyst_agent import build_boc_research_predictor
from boc_rate_decisions.data import build_boc_service


service = build_boc_service(
    reports_dir=Path("data/reports/boc_press_releases"),
)

predictor = build_boc_research_predictor(
    max_documents=3,
    max_chars_per_document=6_000,
)
```

If `reports_dir` is omitted, forecast contexts contain no documents. The
research predictor still runs, but its `research_evidence` list is empty.
The executable notebook and CLI resolve this directory from the repository
root, so they also work when the kernel or shell uses another current working
directory.

## Inspecting cutoff behavior

Test document selection before making an LLM call:

```python
from datetime import datetime

from boc_rate_decisions.research import format_research_evidence


origin = datetime(2024, 5, 8)
context = service.context(origin)

evidence = format_research_evidence(
    context,
    max_documents=3,
    max_chars_per_document=2_000,
)

for item in evidence:
    print(item["publication_date"], item["document_id"])

assert all(item["publication_date"] <= "2024-05-08" for item in evidence)
```

For this origin, a release from the June 2024 meeting must not appear.

## Inspecting the prompt without calling a model

```python
import json

import yaml
from aieng.forecasting.evaluation import BacktestSpec
from boc_rate_decisions.analyst_agent import BoCDecisionPromptBuilder
from boc_rate_decisions.research import DEFAULT_RESEARCH_SOURCES


with open(
    "implementations/boc_rate_decisions/specs/boc_rate_direction_smoke.yaml"
) as file:
    spec = BacktestSpec.model_validate(yaml.safe_load(file))

builder = BoCDecisionPromptBuilder(
    document_sources=DEFAULT_RESEARCH_SOURCES,
    max_documents=3,
    max_chars_per_document=2_000,
)

payload = json.loads(builder(task=spec.task, context=context))

print(json.dumps(payload["research_evidence"], indent=2))
```

The payload also contains the task, origin, announcement date, policy-rate
history, past meeting outcomes, and leak-safe macro snapshot.

## Running an actual forecast

```python
prediction = predictor.predict(
    task=spec.task,
    context=context,
)

print(prediction.payload)
print(prediction.metadata)
```

The payload contains normalized probabilities for `cut`, `hold`, and `hike`.
The metadata contains the agent rationale and key signals used by the
reasoning-alignment evaluator.

An actual forecast requires the repository's configured model credentials and
incurs an LLM call.

## Automated tests

Run the focused research tests:

```bash
uv run pytest implementations/tests/boc_rate_decisions/test_research.py -v
```

Run all BoC tests:

```bash
uv run pytest implementations/tests/boc_rate_decisions -v
```

Run the repository quality checks before pushing:

```bash
make lint
```

The focused tests verify:

- documents after the origin remain hidden;
- documents from multiple sources are merged deterministically;
- only the newest configured number of documents is selected;
- evidence carries source and publication provenance;
- text is truncated at the configured character budget;
- a zero document budget returns no evidence.

## Notebook versus script

Use [`04_research_grounded_forecast.ipynb`](04_research_grounded_forecast.ipynb)
while developing the method. It displays retrieved evidence and the exact
prompt before an opt-in live call, then renders probabilities, rationale, and
key signals as Markdown. Its second half runs a three-origin comparison against
historical frequencies, logistic regression, a clearly labeled two-year-yield
market proxy, and optional quantitative-only/research-grounded agent backtests.
The notebook-level `AGENT_MODEL` setting controls both the single live forecast
and the two model-backed comparison rows; it defaults to
`gemini-3.1-flash-lite-preview`, with `gemini-3.5-flash` shown as the advanced
alternative.
The proxy is diagnostic: it must be replaced with historical meeting-specific
CORRA/OIS probabilities before drawing conclusions about market-relative skill.

Use a script once the experiment is stable and needs repeatable batch execution,
caching, or CI. Unit and integration assertions should remain in pytest rather
than notebook cells.

The matching CLI displays evidence and the prompt without calling a model:

```bash
uv run python implementations/boc_rate_decisions/run_research_agent.py
```

Add `--live` only when you intend to invoke the configured model:

```bash
uv run python implementations/boc_rate_decisions/run_research_agent.py --live
```

The CLI defaults to `gemini-3.1-flash-lite-preview`. Select the advanced model
explicitly with:

```bash
uv run python implementations/boc_rate_decisions/run_research_agent.py \
  --model gemini-3.5-flash --live
```

Run the deterministic three-origin comparison with:

```bash
uv run python implementations/boc_rate_decisions/run_research_agent.py --compare
```

Add both LLM agents—six calls using the selected model—with:

```bash
uv run python implementations/boc_rate_decisions/run_research_agent.py \
  --compare --include-agents --model gemini-3.1-flash-lite-preview
```

The CLI comparison prints the RPS leaderboard and a predicted-versus-actual
table containing confidence, the three category probabilities, correctness,
and per-meeting RPS.

## Adding another research source

To add Monetary Policy Reports, surveys, speeches, or deliberation summaries:

1. Cache each document as an `ExtractedDocument` artifact with its real public
   release date.
2. Load its directory into `DocumentStore` under a distinct source key.
3. Add that key to the prompt builder's `document_sources` tuple.
4. Add a cutoff test containing one visible and one future document.
5. Compare the new source against the press-release-only predictor as an
   ablation before including it in the default method.

Do not fetch live documents from inside a historical prediction. Acquisition
and forecasting must remain separate so historical runs are reproducible and
cutoff enforcement stays inspectable.
