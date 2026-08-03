# Databricks notebook source
# MAGIC %md
# MAGIC # Execution-Trace Checker Comparison — custom_rag vs. custom_rag_v2 vs. custom_rag_v2_checked
# MAGIC
# MAGIC Rebuilds the index (picks up any pending knowledge-base-upload/ doc
# MAGIC changes sitting in the Volume) and runs all three hard-scenario
# MAGIC benchmark arms. Production is deliberately NOT one of them anymore —
# MAGIC across every run so far the custom RAG pipeline has consistently and
# MAGIC clearly exceeded it, so this isolates differences BETWEEN the custom
# MAGIC pipeline variants instead:
# MAGIC
# MAGIC 1. **custom_rag** (v1) — same production prompt + plain top-K retrieval,
# MAGIC    Sonnet generation.
# MAGIC 2. **custom_rag_v2** — same as (1), plus a post-retrieval relevance
# MAGIC    filter (Haiku) and an explicit grounding note, both modeled on
# MAGIC    Ibotta's own internal HR Bot's validated production source.
# MAGIC 3. **custom_rag_v2_checked** — same as (2), plus a narrow, cheap-model
# MAGIC    (Haiku) execution-trace checker wired directly into the self-repair
# MAGIC    loop itself (see `execution_checker.py`) — not just a post-hoc
# MAGIC    measurement. Scoped to exactly one question: would this workflow's
# MAGIC    cross-node references, approval/self-loop guards, and Merge/
# MAGIC    synchronization points actually behave correctly if traced step by
# MAGIC    step? If it finds something, that feeds back as a real repair turn
# MAGIC    before the workflow is ever accepted as final.
# MAGIC
# MAGIC **Cost/latency note:** arm 3 adds a Haiku call on every turn that passes
# MAGIC structural validation (not just failures), plus a full extra repair-turn
# MAGIC call whenever it finds something — expect this arm to run slower and use
# MAGIC more tokens than plain v2. That's the tradeoff being measured here: does
# MAGIC the extra cost actually buy a measurable drop in blockers?
# MAGIC
# MAGIC A re-embed is included as standard practice (safe to re-run any time the
# MAGIC docs/examples corpus changes) — same delete-and-recreate flow as
# MAGIC `rag_v2_comparison.py`, since `index.sync()` doesn't pick up a schema/
# MAGIC content change of this kind.

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

try:
    index = _create_index()
    print(f"Created index: {INDEX_NAME}")
except Exception as e:
    if "already exists" in str(e).lower() or "RESOURCE_ALREADY_EXISTS" in str(e):
        print(f"Index {INDEX_NAME} already exists — deleting and recreating so it picks up "
              f"the current chunk boundaries/schema (re-embeds from scratch).")
        client.delete_index(endpoint_name=ENDPOINT_NAME, index_name=INDEX_NAME)
        index = _create_index()
        print(f"Recreated index: {INDEX_NAME}")
    else:
        raise

# COMMAND ----------

# MAGIC %md ### Wait for the index to come online

# COMMAND ----------

import time

while True:
    state = index.describe().get("status", {}).get("detailed_state", "")
    if state.startswith("ONLINE"):
        print("Index is ONLINE")
        break
    print(f"Waiting for index to be ONLINE (currently: {state or 'unknown'})…")
    time.sleep(15)

# COMMAND ----------

# MAGIC %md ### Smoke test — confirm the index is actually answering before benchmarking

# COMMAND ----------

_smoke = index.similarity_search(
    query_text="how do I avoid a Slack bot replying to its own messages",
    columns=["id", "title", "text", "source"],
    num_results=3,
)
for r in _smoke["result"]["data_array"]:
    print(r[0], "-", r[1])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 2 — Benchmark: custom_rag vs. custom_rag_v2 vs. custom_rag_v2_checked
# MAGIC `benchmark.run_hard()` already runs all three arms and prints both
# MAGIC qualitative win/loss comparisons (custom_rag-vs-custom_rag_v2 and
# MAGIC custom_rag_v2-vs-custom_rag_v2_checked) — no separate wiring needed here.

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

# MAGIC %md ### Inspect every custom_rag_v2_checked failure/warning/soundness issue
# MAGIC This is the new arm — the one that actually reflects the execution-trace
# MAGIC checker's changes. Worth comparing directly against the custom_rag_v2
# MAGIC section below on the SAME scenarios to see what, if anything, the checker
# MAGIC actually fixed.

# COMMAND ----------

for r in results["custom_rag_v2_checked"]:
    if not r.structural.valid or r.structural.warnings or r.soundness_issues:
        print("Scenario:", r.input.category)
        print("Structural errors:", r.structural.errors)
        print("Warnings (Layer 2):", r.structural.warnings)
        print("Soundness issues (Layer 3):", r.soundness_issues)
        print("Actual response:", r.actual_response[:2000])
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

# MAGIC %md ### Inspect every custom_rag (v1) failure/warning/soundness issue, for reference

# COMMAND ----------

for r in results["custom_rag"]:
    if not r.structural.valid or r.structural.warnings or r.soundness_issues:
        print("Scenario:", r.input.category)
        print("Structural errors:", r.structural.errors)
        print("Warnings (Layer 2):", r.structural.warnings)
        print("Soundness issues (Layer 3):", r.soundness_issues)
        print("Actual response:", r.actual_response[:2000])
        print("---")

# COMMAND ----------

# MAGIC %md ### Did the checker actually change the outcome on any shared scenario?
# MAGIC Scenario-by-scenario diff of blocker counts between custom_rag_v2 and
# MAGIC custom_rag_v2_checked — the direct answer to "did this help."

# COMMAND ----------

v2_by_scenario = {r.input.category: r for r in results["custom_rag_v2"]}
checked_by_scenario = {r.input.category: r for r in results["custom_rag_v2_checked"]}

improved, regressed, unchanged = [], [], []
for scenario, v2_r in v2_by_scenario.items():
    checked_r = checked_by_scenario.get(scenario)
    if checked_r is None:
        continue
    v2_blockers = len(v2_r.soundness_blockers)
    checked_blockers = len(checked_r.soundness_blockers)
    if checked_blockers < v2_blockers:
        improved.append((scenario, v2_blockers, checked_blockers))
    elif checked_blockers > v2_blockers:
        regressed.append((scenario, v2_blockers, checked_blockers))
    else:
        unchanged.append((scenario, v2_blockers, checked_blockers))

print(f"Improved (fewer blockers with checker): {len(improved)}")
for s, before, after in improved:
    print(f"  {s}: {before} -> {after} blockers")

print(f"\nRegressed (more blockers with checker): {len(regressed)}")
for s, before, after in regressed:
    print(f"  {s}: {before} -> {after} blockers")

print(f"\nUnchanged: {len(unchanged)}")
for s, before, after in unchanged:
    print(f"  {s}: {before} blockers (both arms)")

# COMMAND ----------

# MAGIC %md ### Which arms reach for an unavailable provider (OpenAI)?
# MAGIC Direct per-scenario check — OpenAI is not an available credential on
# MAGIC this platform, so any hit is a hallucinated integration.

# COMMAND ----------

_MARKERS = ("lmchatopenai", "nodes-langchain.openai", "nodes-base.openai", "gpt-4")
for arm in ("custom_rag", "custom_rag_v2", "custom_rag_v2_checked"):
    hits = [
        r.input.category for r in results[arm]
        if any(m in (r.actual_response or "").lower() for m in _MARKERS)
    ]
    print(f"{arm}: {len(hits)} scenario(s) using OpenAI -> {hits}")
