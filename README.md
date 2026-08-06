# n8n Workflow Builder — RAG pipeline and eval harness

Builds n8n workflow JSON from a plain-English Slack request, and — the larger
half of this repo — measures whether what it built is actually correct.

The generator is the easy part. Getting a model to emit *plausible* n8n JSON
takes an afternoon. Nearly all the work here went into telling the difference
between a workflow that imports cleanly and one that does what was asked,
because those are very different things and only the first is easy to check.

---

## What it does

A Slack thread describing an automation goes in. Out comes either a finished
workflow JSON file or a clarifying question.

```
Slack thread
    │
    ▼
scoping agent ─── not specific enough ──▶ ask the user
    │ ready
    ▼
credential resolution   ← which credentials THIS person can actually reach
    │
    ▼
┌─────────────── one Databricks Model Serving endpoint ───────────────┐
│  retrieve  →  relevance filter  →  generate  →  validate  →  repair │
│                                                     ▲          │    │
│                                                     └──────────┘    │
│                                                    up to 3 rounds   │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
workflow JSON  or  a question for the user
```

Everything inside that box used to be separate n8n nodes — builder, validator,
retry loop, checker, fixer. It is now one call. The reason is the repair loop:
it is only useful if the thing that finds a defect can hand it straight back to
the thing that can fix it, *with the docs for that specific defect still in
hand*. Across a service boundary that round trip is somebody else's
orchestration problem; inside one call it is a `for` loop.

The consolidation has a second effect that matters more. The benchmark and
production now run the **same code path**, so a benchmark number finally
predicts production behaviour instead of describing a harness nobody ships.

---

## Validation: three layers

Ordered by how much you can trust them.

**Layer 1 — deterministic structural check.** Does it parse, is every node type
real, does every connection name a node that exists, is every referenced node
reachable. No LLM. Cannot be argued with.

**Layer 2 — schema-parameter check** (`schema_check.py`). Every parameter on
every node checked against n8n's own `nodes.json` manifest — the same file the
n8n editor uses — so an invented parameter is caught by definition rather than
by a model's opinion. This is a pure-Python port of `check_params.js`, asserted
equivalent to it on 16 workflows including the 65-node production builder
(`n8n_schema_check/equivalence_check.py`). It was ported because the original
shelled out to Node, which would have meant a Node runtime plus `node_modules`
inside the serving image and a process spawn per request — and because when
`node_modules` was absent the check silently reported "unavailable" instead of
failing.

**Layer 3 — LLM judge** (`judge.py`). The only layer that can assess whether the
workflow does what was *asked*. Also the only layer that can be wrong, which is
why most of that file is a list of **verified facts** — things confirmed by
executing real workflows on a live n8n instance rather than by reasoning. Each
one exists because the judge previously flagged correct behaviour as a bug.

That last point generalises: a large share of apparent generation failures were
measurement failures. The judge confidently reporting a non-existent bug and the
generator producing a real one look identical in a score.

---

## Layout

### Run these

| File | What it does |
|---|---|
| `refresh_and_deploy.py` | The one to run. Refreshes the knowledge base, then deploys the endpoint. |
| `rag_setup.py` | Chunks docs from the UC volume, rebuilds the vector index. |
| `deploy_workflow_builder_endpoint.py` | Logs the pipeline as an MLflow model, creates/updates the serving endpoint. |
| `model_comparison_benchmark.py` | Two arms differing by exactly one variable — the generation model. |
| `workflow_builder_eval.py` | The prompt-optimization loop. |
| `optimize.py` | Same loop, as a local CLI. |
| `revalidate_cached_conversations.py` | Re-scores cached conversations after a validator change. Zero tokens. |

### `prompt_optimizer/`

**Serving path** — what production actually runs:
`serving.py` (the pipeline), `serving_model.py` (MLflow wrapper),
`rag_retriever.py`, `relevance_filter.py`, `rag_pipeline_v2.py`,
`query_rewriter.py`, `validator.py`, `schema_check.py`, `llm_response.py`.

**Eval path** — everything used to measure it:
`evaluator.py`, `judge.py`, `benchmark.py`, `synthetic_data.py`,
`hard_scenarios.py`, `loop.py`, `optimizer.py`, `tracker.py`.

`config.py`, `kb_chunker.py`, `n8n_client.py` support both.

---

## Design decisions worth knowing

**Nothing about the environment is hardcoded in the prompt.** Credential IDs are
resolved per request from the live instance and passed in. Generated workflows
get imported into other projects, so a baked-in ID produces a workflow that
looks importable and then fails at runtime pointing at a credential the
importing project cannot see. A *missing* credential is a ten-second fix after
import; a *wrong* one is a support ticket. The prompt therefore instructs the
model to leave `credentials` unset and say so, rather than guess.

**`config.yaml` is logged as a model artifact, not read at request time.** The
prompt text and retrieval settings are pinned to a model version, so the
deployed system cannot drift from the configuration that was benchmarked, and
"which prompt produced this workflow" has an answer.

**The endpoint credential is a short-lived passthrough token,** minted by Model
Serving from the `resources=` declared at log time. No secret scope, no PAT to
rotate, and the credential is scoped to exactly the resources the model calls.

**A clarifying question is returned, not answered.** In the benchmark a
simulated user answers so runs can proceed unattended. In production that would
invent an answer on the user's behalf and build against it, so a non-JSON reply
comes back as `status="question"` for n8n to relay. Repair turns are the
opposite case — the machine talking to itself about something it can verify — so
they stay internal.

**Index readiness means "a query works", not "status says ONLINE".** There is a
window where the index reports online and still rejects searches. Polling the
real operation removes the guess.

**The eval cache key hashes the source of every file that affects a
conversation's outcome.** Change the validator, the retriever, or the response
parser and stale cached results invalidate themselves. This exists because three
consecutive bug fixes to the schema checker changed nothing in the measured
scores — it was invoked by subprocess and therefore not hashed.

---

## Setup

```bash
pip install -r requirements.txt
```

Needs `DATABRICKS_HOST` and `DATABRICKS_TOKEN` in the environment; `config.yaml`
resolves `${VAR}` references from there. On Databricks the notebooks take the
running user's token, which is why they need no configuration — and why the
serving endpoint, which has no user, needs `resources=` instead.

Knowledge base docs live in a Unity Catalog volume, not in this repo. Uploading
them is not enough on its own: the index reads a Delta table, so `rag_setup.py`
has to re-chunk before anything is visible to retrieval.

---

## Limitations

- The judge's verified-facts list covers the node behaviours that have actually
  been tested on a live instance. It is not exhaustive, and an unlisted node's
  runtime behaviour is still guessed at.
- Benchmarking is on hand-written hard scenarios, not production traffic.
- Credential matching for personal accounts is name-based, so an unusually named
  credential can be missed.
- The n8n-side workflow assumes a specific Slack app and instance layout; it is
  not portable as-is.
