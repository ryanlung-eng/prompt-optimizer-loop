# Databricks notebook source
# MAGIC %md
# MAGIC # RAG v2 Comparison — production vs. custom_rag vs. custom_rag_v2
# MAGIC
# MAGIC Rebuilds the index with the latest chunking fix (the merge-forward bug
# MAGIC that glued small sections onto unrelated large ones, e.g. "Credential
# MAGIC Types" onto "Google Sheets Trigger") and runs all three hard-scenario
# MAGIC benchmark arms:
# MAGIC
# MAGIC 1. **production** — the KA endpoint, unchanged.
# MAGIC 2. **custom_rag** — same production prompt + plain top-K retrieval
# MAGIC    (Sonnet generation, now fixed from the earlier Haiku default bug).
# MAGIC 3. **custom_rag_v2** — same as (2), plus a post-retrieval relevance
# MAGIC    filter (Haiku) that drops chunks that don't actually help the
# MAGIC    specific request, and an explicit grounding note — both modeled on
# MAGIC    Ibotta's own internal HR Bot's validated production source.
# MAGIC
# MAGIC A re-embed is required regardless of the new arm, since the chunk
# MAGIC boundaries themselves changed (merge-bug fix) — this notebook rebuilds
# MAGIC the source table + recreates the index (same delete-and-recreate flow
# MAGIC as `rag_setup_and_benchmark.py`, since `index.sync()` doesn't pick up a
# MAGIC schema/content change of this kind) before benchmarking.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pull the latest commit into this workspace
# MAGIC Best-effort — updates the Databricks Repo backing this path to the tip of
# MAGIC `main` via the Repos API, so the cells below actually import the latest
# MAGIC `evaluator.py`/`benchmark.py`/`relevance_filter.py`/`rag_pipeline_v2.py`
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
# MAGIC Required because the chunk-merge bug fix changes chunk boundaries again
# MAGIC (not just because of the new arm) — safe to re-run any time the chunker
# MAGIC or the docs/examples corpus changes.

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

# Do NOT accept any state merely starting with "ONLINE": the index reports
# ONLINE_INITIAL_UPDATE while the first embedding sync is still running, and
# exiting on that made the smoke test below fail with embeddings-not-ready
# errors on every first run (observed live, twice). Fully synced means
# detailed_state == ONLINE_NO_PENDING_UPDATE, plus the indexed row count
# (when the API reports one) covering everything we just wrote.
while True:
    status = index.describe().get("status", {})
    state = status.get("detailed_state", "")
    ready = status.get("ready", False)
    n_indexed = status.get("indexed_row_count")
    count_ok = n_indexed is None or int(n_indexed) >= len(chunks)
    if ready and state == "ONLINE_NO_PENDING_UPDATE" and count_ok:
        print(f"Index is fully online: {state}, indexed_row_count={n_indexed}")
        break
    print(f"Waiting for full sync (state={state or 'unknown'}, ready={ready}, "
          f"indexed={n_indexed}/{len(chunks)})…")
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
# MAGIC ## Phase 2 — Benchmark: production vs. custom_rag vs. custom_rag_v2
# MAGIC `benchmark.run_hard()` already runs all three arms and prints both
# MAGIC qualitative win/loss comparisons (production-vs-custom_rag and
# MAGIC custom_rag-vs-custom_rag_v2) — no separate wiring needed here.

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

# MAGIC %md ### Inspect every custom_rag_v2 failure/warning/soundness issue in detail
# MAGIC This is the new arm — the one that actually reflects this session's
# MAGIC changes (relevance filter + grounding note on top of the Sonnet fix).

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

# MAGIC %md ### Inspect PRODUCTION's failures too
# MAGIC Previously never printed, so production's outputs were never actually
# MAGIC inspected — "production doesn't have problem X" was an assumption, not
# MAGIC an observation. It scores LOWEST of the three arms on knowledge_honesty,
# MAGIC so it's worth looking at directly rather than assuming it's clean.

# COMMAND ----------

for r in results["production"]:
    if not r.structural.valid or r.structural.warnings or r.soundness_issues:
        print("Scenario:", r.input.category)
        print("Structural errors:", r.structural.errors)
        print("Warnings (Layer 2):", r.structural.warnings)
        print("Soundness issues (Layer 3):", r.soundness_issues)
        print("Actual response:", r.actual_response[:2000])
        print("---")

# COMMAND ----------

# MAGIC %md ### Which arms reach for an unavailable provider (OpenAI)?
# MAGIC Direct per-scenario check — OpenAI is not an available credential on
# MAGIC this platform, so any hit is a hallucinated integration.

# COMMAND ----------

_MARKERS = ("lmchatopenai", "nodes-langchain.openai", "nodes-base.openai", "gpt-4")
for arm in ("production", "custom_rag", "custom_rag_v2"):
    hits = [
        r.input.category for r in results[arm]
        if any(m in (r.actual_response or "").lower() for m in _MARKERS)
    ]
    print(f"{arm}: {len(hits)} scenario(s) using OpenAI -> {hits}")
