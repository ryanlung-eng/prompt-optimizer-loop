# Databricks notebook source
# MAGIC %md
# MAGIC # RAG Setup + Hard-Scenario Benchmark
# MAGIC
# MAGIC One notebook, two phases:
# MAGIC
# MAGIC 1. **Setup** (same as `rag_setup.py`) — chunks the `docs/` and
# MAGIC    `examples/` folders in the existing `dev.platform.automation-builder`
# MAGIC    Unity Catalog volume, structure-aware: each file splits on its own
# MAGIC    internal `## ` headers (one node/topic per chunk) when it has any,
# MAGIC    rather than always embedding a whole multi-topic file as one vector.
# MAGIC    Writes to a Delta table, and creates the Databricks AI Search endpoint
# MAGIC    + Delta Sync index over it. Safe to re-run — creation is skipped (and
# MAGIC    a sync triggered instead) if the endpoint/index already exist.
# MAGIC 2. **Benchmark** (same as `benchmark_rag_vs_ka.py`) — compares the
# MAGIC    production Knowledge Assistant endpoint against the new custom RAG
# MAGIC    pipeline (frozen `instructions.md` + retrieval over the index from
# MAGIC    phase 1) on the 18 Layer 4 hard scenarios, scored by the same
# MAGIC    judge/validator as the rest of the eval pipeline.
# MAGIC
# MAGIC `instructions.md` itself is NOT indexed — at ~23.5k tokens it's small
# MAGIC enough to always-inject in full rather than retrieve from; only the
# MAGIC docs/examples corpus (too big to always-inject) goes through retrieval.
# MAGIC See `evaluator.py`'s `run_batch_custom_rag()` for how the two combine.

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

# MAGIC %pip install httpx tenacity rich nest_asyncio pyyaml mlflow databricks-ai-search -q
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
# MAGIC ## Phase 1 — Setup: build the source table, endpoint, and index

# COMMAND ----------

from prompt_optimizer.kb_chunker import chunk_directory_by_file

# dev.platform already exists (and the docs are already uploaded to the
# automation-builder volume there) — no CREATE CATALOG/SCHEMA needed, and no
# permission for it anyway. Table name is scoped with the automation_builder_
# prefix since platform is a shared schema other tables may already live in.
CATALOG = "dev"
SCHEMA = "platform"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.automation_builder_kb_chunks"
ENDPOINT_NAME = "n8n-kb-endpoint"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.automation_builder_kb_chunks_index"
EMBEDDING_MODEL_ENDPOINT = "databricks-gte-large-en"
DOCS_DIR = "/Volumes/dev/platform/automation-builder/docs"
EXAMPLES_DIR = "/Volumes/dev/platform/automation-builder/examples"

# COMMAND ----------

# Merged into one table/index — id_prefix keeps the two sources' IDs from
# colliding (both would otherwise restart numbering at 000).
doc_chunks = chunk_directory_by_file(DOCS_DIR, id_prefix="docs")
example_chunks = chunk_directory_by_file(EXAMPLES_DIR, id_prefix="examples")
chunks = doc_chunks + example_chunks
print(f"{len(doc_chunks)} chunks from {DOCS_DIR}, {len(example_chunks)} from {EXAMPLES_DIR}, {len(chunks)} total")

rows = [c.to_dict() for c in chunks]
df = spark.createDataFrame(rows)

# Delta Sync indexes require Change Data Feed enabled on the source table.
(
    df.write.format("delta")
    .mode("overwrite")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(SOURCE_TABLE)
)
print(f"Wrote {df.count()} rows to {SOURCE_TABLE}")

# COMMAND ----------

from databricks.ai_search.client import AISearchClient

client = AISearchClient()

# Endpoint already exists and is valid — skipping recreation. Uncomment if
# you ever need to provision a fresh one (e.g. a new ENDPOINT_NAME).
# try:
#     client.create_endpoint(name=ENDPOINT_NAME, endpoint_type="STANDARD")
#     print(f"Created endpoint: {ENDPOINT_NAME}")
# except Exception as e:
#     if "already exists" in str(e).lower() or "RESOURCE_ALREADY_EXISTS" in str(e):
#         print(f"Endpoint {ENDPOINT_NAME} already exists, skipping.")
#     else:
#         raise

# COMMAND ----------

try:
    index = client.create_delta_sync_index(
        endpoint_name=ENDPOINT_NAME,
        source_table_name=SOURCE_TABLE,
        index_name=INDEX_NAME,
        pipeline_type="TRIGGERED",   # sync manually via index.sync() after re-running the cell above
        primary_key="id",
        embedding_source_column="text",
        embedding_model_endpoint_name=EMBEDDING_MODEL_ENDPOINT,
    )
    print(f"Created index: {INDEX_NAME}")
except Exception as e:
    if "already exists" in str(e).lower() or "RESOURCE_ALREADY_EXISTS" in str(e):
        print(f"Index {INDEX_NAME} already exists — fetching it and triggering a sync instead.")
        index = client.get_index(index_name=INDEX_NAME)
        index.sync()
    else:
        raise

# COMMAND ----------

# MAGIC %md ### Wait for the index to come online
# MAGIC A freshly created index needs to provision the endpoint compute and embed
# MAGIC every row before it's queryable — Databricks' own docs say to expect this
# MAGIC to take several minutes. `BadRequest: ... is not ready` right after
# MAGIC creation is this, not a real error; this cell polls until it's done
# MAGIC instead of guessing when to re-run.

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

# MAGIC %md ### Smoke test — confirm the index is actually answering before benchmarking against it

# COMMAND ----------

_smoke = index.similarity_search(
    query_text="how do I avoid a Slack bot replying to its own messages",
    columns=["id", "title", "text"],
    num_results=3,
)
for r in _smoke["result"]["data_array"]:
    print(r[0], "-", r[1])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 2 — Benchmark: production KA vs. custom RAG on the hard scenarios

# COMMAND ----------

from prompt_optimizer.config import load_config
from prompt_optimizer import benchmark

cfg = load_config("/Workspace/Users/ryan.lung@ibotta.com/prompt-optimizer-loop/config.yaml")

results = asyncio.get_event_loop().run_until_complete(benchmark.run_hard(cfg))

# COMMAND ----------

# MAGIC %md ### Inspect every custom_rag failure/warning/soundness issue in detail

# COMMAND ----------

for r in results["custom_rag"]:
    if not r.structural.valid or r.structural.warnings or r.soundness_issues:
        print("Scenario:", r.input.category)
        print("Structural errors:", r.structural.errors)
        print("Warnings (Layer 2):", r.structural.warnings)
        print("Soundness issues (Layer 3):", r.soundness_issues)
        print("Actual response:", r.actual_response[:2000])
        print("---")
