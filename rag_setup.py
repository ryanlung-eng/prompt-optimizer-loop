# Databricks notebook source
# MAGIC %md
# MAGIC # RAG Setup — n8n Knowledge Base
# MAGIC
# MAGIC One-time (or re-run-when-docs-change) setup notebook. Chunks
# MAGIC `instructions-digest.md` by section, writes it to a Unity Catalog Delta
# MAGIC table, then creates a Databricks AI Search endpoint + Delta Sync index over
# MAGIC it. After this runs, the index is a managed, always-on resource — this
# MAGIC notebook does NOT need to keep running. Query it at request time via
# MAGIC `prompt_optimizer/rag_retriever.py`.
# MAGIC
# MAGIC Re-run this notebook (from the "Build source table" cell onward) whenever
# MAGIC `instructions-digest.md` changes — the Delta Sync index picks up table
# MAGIC changes automatically, no separate re-index step needed.

# COMMAND ----------

# MAGIC %pip install databricks-ai-search -q
# MAGIC %pip install --upgrade typing_extensions -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import sys

sys.path.insert(0, "/Workspace/Users/ryan.lung@ibotta.com/prompt-optimizer-loop")

from prompt_optimizer.kb_chunker import load_and_chunk

# COMMAND ----------

# MAGIC %md ## Config — change these if you rename the catalog/schema/endpoint

# COMMAND ----------

CATALOG = "main"
SCHEMA = "n8n_kb"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.doc_chunks"
ENDPOINT_NAME = "n8n-kb-endpoint"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.doc_chunks_index"
EMBEDDING_MODEL_ENDPOINT = "databricks-gte-large-en"
DOC_PATH = "/Workspace/Users/ryan.lung@ibotta.com/prompt-optimizer-loop/instructions-digest.md"

# COMMAND ----------

# MAGIC %md ## Build source table

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

chunks = load_and_chunk(DOC_PATH)
print(f"{len(chunks)} chunks parsed from {DOC_PATH}")

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
