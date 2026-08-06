# Databricks notebook source
# MAGIC %md
# MAGIC # Sonnet 4.6 vs Opus 5 — identical RAG pipeline, identical prompt
# MAGIC
# MAGIC Rebuilds the index (picks up any pending knowledge-base-upload/ doc
# MAGIC changes sitting in the Volume) and runs two hard-scenario arms that
# MAGIC differ by exactly ONE variable — the generation model:
# MAGIC
# MAGIC 1. **custom_rag_v2** — Sonnet 4.6, with the post-retrieval relevance
# MAGIC    filter (Haiku) and grounding note, both modeled on Ibotta's own
# MAGIC    internal HR Bot's validated production source.
# MAGIC 2. **custom_rag_v2_strong** — the SAME pipeline and the SAME prompt,
# MAGIC    with only the generation model swapped to Opus 5. This is the
# MAGIC    cleanest comparison this benchmark has had: one variable.
# MAGIC    It answers how much of the residual blocker count is a model
# MAGIC    ceiling rather than a scaffolding gap — which matters because the
# MAGIC    two in-loop mechanisms tried so far (execution checker, state
# MAGIC    simulation) both looked principled and both measured negative.
# MAGIC
# MAGIC    Note Opus 5 rejects the `temperature` parameter, so `_call` drops it
# MAGIC    and retries; that arm therefore samples at the endpoint default
# MAGIC    rather than temperature=0 and will vary more run to run.
# MAGIC    It is also serialized with a per-call pause for a workspace
# MAGIC    tokens-per-minute quota, so it is SLOW — roughly an hour.
# MAGIC
# MAGIC **Dropped arm — `custom_rag_v2_statesim`**: its first run looked like a
# MAGIC win only because two workflows came back structurally INVALID (the
# MAGIC instruction made the model narrate its simulation instead of emitting
# MAGIC JSON), which left the reviewer nothing to find. With that fixed
# MAGIC (17/17 valid) it measured clearly worse than plain v2 — 21 blockers vs
# MAGIC 14, completeness 0.868 vs 0.926.
# MAGIC
# MAGIC **Dropped arm — `custom_rag_v2_checked`** (in-loop execution-trace
# MAGIC checker): retired after three runs. It consistently scored WORST overall
# MAGIC despite having the fewest blockers — its repair turns traded completeness
# MAGIC for blocker-avoidance (completeness 0.67 vs plain v2's 0.88 on the last
# MAGIC run). Trace analysis showed every repair turn issuing at least one false
# MAGIC demand: a nonexistent double-wrapped output path
# MAGIC (`$json.output.output.field`), and re-routing the approval DM away from
# MAGIC the workflow owner. The `execution_check=True` toggle still exists in
# MAGIC `evaluator.py` if it's ever worth revisiting with a stronger checker model.
# MAGIC
# MAGIC A re-embed is included as standard practice (safe to re-run any time the
# MAGIC docs/examples corpus changes) — the same delete-and-recreate flow
# MAGIC `rag_setup.py` uses, since `index.sync()` does not pick up a schema
# MAGIC change of this kind.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pull the latest commit into this workspace
# MAGIC Best-effort — updates the Databricks Repo backing this path to the tip of
# MAGIC `main` via the Repos API, so the cells below actually import the latest
# MAGIC `evaluator.py`/`benchmark.py`/`execution_checker.py`/`rag_pipeline_v2.py`
# MAGIC instead of a stale cached checkout. If this cell errors (e.g. the path
# MAGIC isn't a Repo, or a permissions issue), pull manually via the Repos UI's
# MAGIC "Pull" button for `/Workspace/Users/ryan.lung@ibotta.com/prompt-optimizer-loop`
# MAGIC before continuing — everything below assumes the latest code is already there.

# COMMAND ----------

_REPO_PATH = "/Workspace/Users/ryan.lung@ibotta.com/prompt-optimizer-loop"

try:
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    _repo = next((r for r in w.repos.list() if r.path == _REPO_PATH), None)
    if _repo is None:
        print(f"No Databricks Repo found at {_REPO_PATH} — is this a plain Workspace "
              f"folder instead of a git-linked Repo? Pull/sync the latest code manually.")
    else:
        w.repos.update(repo_id=_repo.id, branch="main")
        print(f"Updated {_REPO_PATH} (repo_id={_repo.id}) to latest main.")
except Exception as e:
    print(f"Could not auto-pull — pull manually via the Repos UI before continuing. Error: {e}")

# COMMAND ----------

# MAGIC %sh
# MAGIC uname -m
# MAGIC mkdir -p /tmp/node22 && cd /tmp/node22
# MAGIC curl -fsSL -o node.tar.gz https://nodejs.org/dist/v22.9.0/node-v22.9.0-linux-x64.tar.gz
# MAGIC tar -xzf node.tar.gz --strip-components=1
# MAGIC ./bin/npm --version
# MAGIC
# MAGIC mkdir -p /tmp/n8n_schema_check_cache
# MAGIC cp /Workspace/Users/ryan.lung@ibotta.com/prompt-optimizer-loop/prompt_optimizer/n8n_schema_check/check_params.js /tmp/n8n_schema_check_cache/
# MAGIC cp /Workspace/Users/ryan.lung@ibotta.com/prompt-optimizer-loop/prompt_optimizer/n8n_schema_check/package.json /tmp/n8n_schema_check_cache/
# MAGIC cd /tmp/n8n_schema_check_cache && /tmp/node22/bin/npm install --ignore-scripts

# COMMAND ----------

# MAGIC %pip install httpx tenacity rich nest_asyncio pyyaml mlflow databricks-ai-search databricks-sdk -q
# MAGIC %pip install --upgrade typing_extensions -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import sys, os, asyncio
import nest_asyncio
nest_asyncio.apply()  # Databricks notebooks have their own event loop — this patches it

sys.path.insert(0, "/Workspace/Users/ryan.lung@ibotta.com/prompt-optimizer-loop")

# COMMAND ----------

# Pull host + token from cluster context — no secret needed
_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
os.environ["DATABRICKS_HOST"]  = "https://" + _ctx.browserHostName().get()
os.environ["DATABRICKS_TOKEN"] = _ctx.apiToken().get()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 1 — Rebuild the source table + recreate the index
# MAGIC Picks up whatever is currently in the Volume — make sure any pending
# MAGIC `knowledge-base-upload/` doc edits have already been uploaded there
# MAGIC before running this, or this run won't reflect them.

# COMMAND ----------

from prompt_optimizer.kb_chunker import chunk_directory_by_file

CATALOG = "dev"
SCHEMA = "platform"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.automation_builder_kb_chunks"
ENDPOINT_NAME = "n8n-kb-endpoint"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.automation_builder_kb_chunks_index"
EMBEDDING_MODEL_ENDPOINT = "databricks-gte-large-en"
DOCS_DIR = "/Volumes/dev/platform/automation-builder/docs"
EXAMPLES_DIR = "/Volumes/dev/platform/automation-builder/examples"

# COMMAND ----------

doc_chunks = chunk_directory_by_file(DOCS_DIR, id_prefix="docs")
example_chunks = chunk_directory_by_file(EXAMPLES_DIR, id_prefix="examples")
chunks = doc_chunks + example_chunks
print(f"{len(doc_chunks)} chunks from {DOCS_DIR}, {len(example_chunks)} from {EXAMPLES_DIR}, {len(chunks)} total")

rows = [c.to_dict() for c in chunks]
df = spark.createDataFrame(rows)

(
    df.write.format("delta")
    .mode("overwrite")
    .option("delta.enableChangeDataFeed", "true")
    .option("overwriteSchema", "true")
    .saveAsTable(SOURCE_TABLE)
)
print(f"Wrote {df.count()} rows to {SOURCE_TABLE}")

# COMMAND ----------

from databricks.ai_search.client import AISearchClient

client = AISearchClient()

try:
    client.create_endpoint(name=ENDPOINT_NAME, endpoint_type="STANDARD")
    print(f"Created endpoint: {ENDPOINT_NAME}")
except Exception as e:
    if "already exists" in str(e).lower() or "RESOURCE_ALREADY_EXISTS" in str(e):
        print(f"Endpoint {ENDPOINT_NAME} already exists, skipping.")
    else:
        raise

# COMMAND ----------

# index.sync() only picks up row-level content changes — NOT a schema/chunk-
# boundary change, so this always deletes + recreates rather than syncing.
#
# The delete is ASYNCHRONOUS: delete_index() returns as soon as the request is
# accepted, not once the index is gone, so recreating immediately crashed
# every single run ("still exists" / not-yet-initialized). Hence: delete, then
# POLL until it is genuinely gone, and only then create — with the create
# itself retried, since it can also transiently fail while the backend
# finishes tearing the old one down.
import time


def _create_index():
    return client.create_delta_sync_index(
        endpoint_name=ENDPOINT_NAME,
        source_table_name=SOURCE_TABLE,
        index_name=INDEX_NAME,
        pipeline_type="TRIGGERED",
        primary_key="id",
        embedding_source_column="text",
        embedding_model_endpoint_name=EMBEDDING_MODEL_ENDPOINT,
    )


def _index_exists():
    """True only if the index both resolves AND answers describe(). A handle
    alone is not proof of existence — describe() is what actually round-trips
    to the backend."""
    try:
        client.get_index(index_name=INDEX_NAME).describe()
        return True
    except Exception:
        return False


def _create_with_retry(attempts=12, pause=15):
    """Create, tolerating the window where the backend still considers the old
    index present. Re-raises the real error once the budget is spent, so a
    genuine misconfiguration still surfaces instead of looping forever."""
    last = None
    for i in range(attempts):
        try:
            return _create_index()
        except Exception as e:
            last = e
            msg = str(e).lower()
            transient = ("already exists" in msg or "resource_already_exists" in msg
                         or "not ready" in msg or "initializ" in msg
                         or "in progress" in msg or "being deleted" in msg)
            if not transient:
                raise
            print(f"  create attempt {i + 1}/{attempts} hit a transient state "
                  f"({str(e)[:110]}) — waiting {pause}s…")
            time.sleep(pause)
    raise last


if _index_exists():
    print(f"Index {INDEX_NAME} already exists — deleting and recreating so it picks up "
          f"the current chunk boundaries/schema (re-embeds from scratch).")
    client.delete_index(endpoint_name=ENDPOINT_NAME, index_name=INDEX_NAME)
    _deadline = time.time() + 600
    while _index_exists():
        if time.time() > _deadline:
            print("  WARNING: index still reported present 10min after delete — "
                  "attempting create anyway.")
            break
        print("  waiting for the delete to finish…")
        time.sleep(10)
    else:
        print("  delete confirmed complete.")

index = _create_with_retry()
print(f"Index ready: {INDEX_NAME}")

# COMMAND ----------

# MAGIC %md ### Wait for the index to come online

# COMMAND ----------

import time

# Deliberately NOT a hard gate on field names we have not verified. An earlier
# version required status["ready"] AND detailed_state == "ONLINE_NO_PENDING_UPDATE"
# AND indexed_row_count >= len(chunks); if any of those keys or literals differ
# on this Databricks version the loop never exits and the notebook looks hung.
# So: print the raw status once so the real shape is visible, treat "no pending
# update" as the signal when it is available, and ALWAYS bound the wait. The
# retrying smoke test below is the real gate — it proves the index answers,
# which is the only thing we actually care about.
_deadline = time.time() + 900
_first = True
while True:
    status = index.describe().get("status", {}) or {}
    if _first:
        print("raw index status keys:", dict(status))
        _first = False
    state = str(status.get("detailed_state", "") or "")
    n_indexed = status.get("indexed_row_count")
    pending = "PENDING" in state.upper()
    if state.upper().startswith("ONLINE") and not pending:
        print(f"Index reports online with no pending update "
              f"(state={state}, indexed_row_count={n_indexed})")
        break
    if time.time() > _deadline:
        print(f"WARNING: gave up waiting after 15min (state={state}, "
              f"indexed_row_count={n_indexed}) — continuing to the smoke test, "
              f"which retries and will surface a real problem if there is one.")
        break
    print(f"Waiting for sync (state={state or 'unknown'}, indexed={n_indexed}/{len(chunks)})…")
    time.sleep(15)

# COMMAND ----------

# MAGIC %md ### Smoke test — confirm the index is actually answering before benchmarking

# COMMAND ----------

# Retried, not one-shot: even after the index reports fully synced, the
# query path can lag it by a little (observed live — the first query after
# "online" failed while the identical query seconds later succeeded). This
# doubles as the warm-up probe so the benchmark never eats that delay.
_deadline = time.time() + 600
_attempt = 0
while True:
    _attempt += 1
    try:
        _smoke = index.similarity_search(
            query_text="how do I avoid a Slack bot replying to its own messages",
            columns=["id", "title", "text", "source"],
            num_results=3,
        )
        break
    except Exception as e:
        if time.time() > _deadline:
            raise
        print(f"Smoke test attempt {_attempt} failed ({str(e)[:140]}) — "
              f"index still warming up, retrying in 20s…")
        time.sleep(20)
for r in _smoke["result"]["data_array"]:
    print(r[0], "-", r[1])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 2 — Benchmark: both arms
# MAGIC `benchmark.run_hard()` already runs both arms and prints qualitative
# MAGIC win/loss comparisons for each pair — no separate wiring needed here.

# COMMAND ----------

from pathlib import Path
p = Path("/Workspace/Users/ryan.lung@ibotta.com/n8n-optimizer-cache/conversation_cache.json")
if p.exists():
    print(f"deleting {p} ({p.stat().st_size:,} bytes)")
    p.unlink()
else:
    print("already gone")

# COMMAND ----------

from prompt_optimizer.config import load_config
from prompt_optimizer import benchmark

cfg = load_config("/Workspace/Users/ryan.lung@ibotta.com/prompt-optimizer-loop/config.yaml")

results = asyncio.get_event_loop().run_until_complete(benchmark.run_hard(cfg))

# COMMAND ----------

# MAGIC %md ### What retrieval actually saw, per scenario
# MAGIC Every v2 arm logs its retrieval decisions into the transcript (and
# MAGIC into MLflow as a root-span input): the retrieval query and which chunk
# MAGIC sources survived filtering. Read this when an arm wins or loses a
# MAGIC scenario — it answers "what did retrieval actually see" directly,
# MAGIC instead of leaving it to be inferred from the final workflow.

# COMMAND ----------

import json as _json

for r in results["custom_rag_v2"]:
    meta = next((t for t in (r.transcript or []) if t.get("role") == "retrieval_meta"), None)
    if not meta:
        continue
    m = _json.loads(meta["content"])
    print("Scenario:", r.input.category)
    print("  original :", r.input.text[:150])
    print("  query    :", m["retrieval_query"][:250])
    print(f"  probe saw {len(m.get('probe_docs', []))} candidate docs; "
          f"kept {m['n_kept']}/{m['n_retrieved']} chunks from: {m['kept_sources']}")
    # Any vendor the platform has no credential for must NEVER appear in a
    # retrieval query. Kept as a standing check: an earlier query-rewriting
    # arm was caught injecting "OpenAI" — a provider with no credential here —
    # into 4 of 17 queries, which pulled the wrong docs entirely.
    _leaked = [v for v in ("OpenAI", "GPT", "Anthropic", "Postgres", "Notion", "Airtable")
               if v.lower() in m["retrieval_query"].lower()]
    if _leaked:
        print(f"  *** VENDOR LEAK IN QUERY: {_leaked} ***")
    print("---")

# COMMAND ----------

# MAGIC %md ### Inspect every custom_rag_v2 (unchecked) failure/warning/soundness issue

# COMMAND ----------

for r in results["custom_rag_v2"]:
    if not r.structural.valid or r.structural.warnings or r.soundness_issues:
        print("Scenario:", r.input.category)
        print("Structural errors:", r.structural.errors)
        print("Warnings (Layer 2):", r.structural.warnings)
        print("Soundness issues (Layer 3):", r.soundness_issues)
        print("Actual response:", r.actual_response[:2000])
        print("---")

# COMMAND ----------

# MAGIC %md ### Inspect every Opus 5 failure/warning/soundness issue

# COMMAND ----------

for r in results["custom_rag_v2_strong"]:
    if not r.structural.valid or r.structural.warnings or r.soundness_issues:
        print("Scenario:", r.input.category)
        print("Structural errors:", r.structural.errors)
        print("Warnings (Layer 2):", r.structural.warnings)
        print("Soundness issues (Layer 3):", r.soundness_issues)
        print("Actual response:", r.actual_response[:2000])
        print("---")

# COMMAND ----------

# MAGIC %md ### Inspect every custom_rag_v2_strong (Opus 5) failure/warning/soundness issue
# MAGIC Compare against the custom_rag_v2 section above on the SAME scenarios:
# MAGIC anything still broken here is a strong candidate for a genuine model
# MAGIC ceiling rather than something more scaffolding would fix.

# COMMAND ----------

for r in results["custom_rag_v2_strong"]:
    if not r.structural.valid or r.structural.warnings or r.soundness_issues:
        print("Scenario:", r.input.category)
        print("Structural errors:", r.structural.errors)
        print("Warnings (Layer 2):", r.structural.warnings)
        print("Soundness issues (Layer 3):", r.soundness_issues)
        print("Actual response:", r.actual_response[:2000])
        print("---")

# COMMAND ----------

# MAGIC %md ### Did either variant actually change the outcome on any shared scenario?
# MAGIC Scenario-by-scenario diff of blocker counts against the custom_rag_v2
# MAGIC baseline — the direct answer to "did this help," for both toggles.

# COMMAND ----------

def diff_against_v2(variant_arm: str):
    v2_by_scenario = {r.input.category: r for r in results["custom_rag_v2"]}
    variant_by_scenario = {r.input.category: r for r in results[variant_arm]}

    improved, regressed, unchanged = [], [], []
    for scenario, v2_r in v2_by_scenario.items():
        variant_r = variant_by_scenario.get(scenario)
        if variant_r is None:
            continue
        v2_blockers = len(v2_r.soundness_blockers)
        variant_blockers = len(variant_r.soundness_blockers)
        if variant_blockers < v2_blockers:
            improved.append((scenario, v2_blockers, variant_blockers))
        elif variant_blockers > v2_blockers:
            regressed.append((scenario, v2_blockers, variant_blockers))
        else:
            unchanged.append((scenario, v2_blockers, variant_blockers))

    print(f"[{variant_arm}] Improved (fewer blockers): {len(improved)}")
    for s, before, after in improved:
        print(f"  {s}: {before} -> {after} blockers")
    print(f"\n[{variant_arm}] Regressed (more blockers): {len(regressed)}")
    for s, before, after in regressed:
        print(f"  {s}: {before} -> {after} blockers")
    print(f"\n[{variant_arm}] Unchanged: {len(unchanged)}")
    for s, before, after in unchanged:
        print(f"  {s}: {before} blockers (both arms)")

diff_against_v2("custom_rag_v2_strong")

# COMMAND ----------

# MAGIC %md ### Which arms reach for an unavailable provider (OpenAI)?
# MAGIC Direct per-scenario check — OpenAI is not an available credential on
# MAGIC this platform, so any hit is a hallucinated integration.

# COMMAND ----------

_MARKERS = ("lmchatopenai", "nodes-langchain.openai", "nodes-base.openai", "gpt-4")
for arm in ("custom_rag_v2", "custom_rag_v2_strong"):
    hits = [
        r.input.category for r in results[arm]
        if any(m in (r.actual_response or "").lower() for m in _MARKERS)
    ]
    print(f"{arm}: {len(hits)} scenario(s) using OpenAI -> {hits}")
