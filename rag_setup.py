# Databricks notebook source
# MAGIC %md
# MAGIC # RAG Setup — n8n Knowledge Base
# MAGIC
# MAGIC One-time (or re-run-when-docs-change) setup notebook. Chunks the
# MAGIC `knowledge-base-upload/` corpus (59 files, ~129k tokens — the large
# MAGIC "official skills" corpus, one chunk per file since each is already a
# MAGIC single topic), writes it to a Unity Catalog Delta table, then creates a
# MAGIC Databricks AI Search endpoint + Delta Sync index over it. After this runs,
# MAGIC the index is a managed, always-on resource — this notebook does NOT need
# MAGIC to keep running. Query it at request time via
# MAGIC `prompt_optimizer/rag_retriever.py`.
# MAGIC
# MAGIC **Deliberately NOT indexing `instructions.md`** — that corpus is only
# MAGIC ~23.5k tokens, small enough to always-inject in full (frozen, cache-
# MAGIC breakpointed) rather than retrieve from. Retrieval only earns its
# MAGIC complexity on a corpus too big to always-inject — see `evaluator.py`'s
# MAGIC `run_batch_custom_rag()` for how the two combine: frozen `instructions.md`
# MAGIC + retrieved chunks from this index.
# MAGIC
# MAGIC Re-run this notebook (from the "Build source table" cell onward) whenever
# MAGIC `knowledge-base-upload/` changes — the Delta Sync index picks up table
# MAGIC changes automatically, no separate re-index step needed.

# COMMAND ----------

# MAGIC %pip install databricks-ai-search -q
# MAGIC %pip install --upgrade typing_extensions -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import sys

sys.path.insert(0, "/Workspace/Users/ryan.lung@ibotta.com/prompt-optimizer-loop")

from prompt_optimizer.kb_chunker import chunk_directory_by_file

# COMMAND ----------

# MAGIC %md ## Config — change these if you rename the catalog/schema/endpoint

# COMMAND ----------

CATALOG = "main"
SCHEMA = "n8n_kb"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.big_corpus_chunks"
ENDPOINT_NAME = "n8n-kb-endpoint"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.big_corpus_chunks_index"
EMBEDDING_MODEL_ENDPOINT = "databricks-gte-large-en"
KB_DIR = "/Workspace/Users/ryan.lung@ibotta.com/prompt-optimizer-loop/knowledge-base-upload"

# COMMAND ----------

# MAGIC %md ## Build source table

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

chunks = chunk_directory_by_file(KB_DIR)
print(f"{len(chunks)} chunks (one per file) parsed from {KB_DIR}")

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

# MAGIC %md ## Create the endpoint (skip if it already exists)

# COMMAND ----------

from databricks.ai_search.client import AISearchClient

client = AISearchClient()

try:
    client.create_endpoint(name=ENDPOINT_NAME, endpoint_type="STANDARD")
    print(f"Created endpoint: {ENDPOINT_NAME}")
except Exception as e:
    # Already exists on a re-run — safe to ignore, surface anything else.
    if "already exists" in str(e).lower() or "RESOURCE_ALREADY_EXISTS" in str(e):
        print(f"Endpoint {ENDPOINT_NAME} already exists, skipping.")
    else:
        raise

# COMMAND ----------

# MAGIC %md ## Create the Delta Sync index (skip if it already exists)

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

# MAGIC %md ## Smoke test

# COMMAND ----------

results = index.similarity_search(
    query_text="how do I avoid a Slack bot replying to its own messages",
    columns=["id", "title", "text"],
    num_results=3,
)
for r in results["result"]["data_array"]:
    print(r[0], "-", r[1])
