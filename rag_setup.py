# Databricks notebook source
# MAGIC %md
# MAGIC # RAG Setup — n8n Knowledge Base
# MAGIC
# MAGIC One-time (or re-run-when-docs-change) setup notebook. Chunks the
# MAGIC `docs/` and `examples/` folders in the `dev.platform.automation-builder`
# MAGIC volume — structure-aware: each file is split on its own internal `## `
# MAGIC headers (one node/topic per chunk) when it has any, falling back to
# MAGIC whole-file only for files with no internal headers. Splitting within
# MAGIC files matters — several files cover 15-20+ distinct node types each, and
# MAGIC embedding the whole file as one vector dilutes it enough that the exact
# MAGIC node a query needs can still miss the top-K. Writes the chunks to a Unity
# MAGIC Catalog Delta table, then creates a Databricks AI Search endpoint + Delta
# MAGIC Sync index over it. After this runs, the index is a managed, always-on
# MAGIC resource — this notebook does NOT need to keep running. Query it at
# MAGIC request time via `prompt_optimizer/rag_retriever.py`.
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

# MAGIC %md ## Build source table

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
# overwriteSchema is required whenever the chunk schema itself changes (e.g.
# the "source" column added for per-document retrieval capping) — mode
# "overwrite" alone only replaces DATA; without this, Delta rejects the
# write with a metadata/schema mismatch rather than silently altering an
# existing table's schema.
(
    df.write.format("delta")
    .mode("overwrite")
    .option("delta.enableChangeDataFeed", "true")
    .option("overwriteSchema", "true")
    .saveAsTable(SOURCE_TABLE)
)
print(f"Wrote {df.count()} rows to {SOURCE_TABLE}")

# COMMAND ----------

# MAGIC %md ## Create the endpoint (skip if it already exists)

# COMMAND ----------

from databricks.ai_search.client import AISearchClient

client = AISearchClient()

# Idempotent — the except branch below already handles the endpoint
# existing from a prior run, so this is safe to leave uncommented/re-run.
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

# MAGIC %md ## Create the Delta Sync index (recreate if the chunk schema changed)
# MAGIC
# MAGIC **Important:** `index.sync()` only picks up row-level content changes in
# MAGIC the source table (inserts/updates/deletes) — it does NOT pick up a
# MAGIC schema change (a column added/removed, like `source` below). Confirmed
# MAGIC via `BadRequest: Requested columns to fetch are not present in index: source`
# MAGIC after a plain sync — the set of queryable columns is fixed at index
# MAGIC creation. Whenever `kb_chunker.py`'s `DocChunk` fields change, the index
# MAGIC must be deleted and recreated, not just synced.

# COMMAND ----------

def _create_index():
    return client.create_delta_sync_index(
        endpoint_name=ENDPOINT_NAME,
        source_table_name=SOURCE_TABLE,
        index_name=INDEX_NAME,
        pipeline_type="TRIGGERED",   # sync manually via index.sync() after re-running the cell above
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
              f"the current chunk schema/columns (this re-embeds everything from scratch; "
              f"a plain sync would silently keep serving the OLD schema instead).")
        client.delete_index(endpoint_name=ENDPOINT_NAME, index_name=INDEX_NAME)
        index = _create_index()
        print(f"Recreated index: {INDEX_NAME}")
    else:
        raise

# COMMAND ----------

# MAGIC %md ## Wait until the index actually answers queries
# MAGIC A freshly created index provisions compute and embeds every row before it
# MAGIC is queryable — expect several minutes.
# MAGIC
# MAGIC **Why this polls a real query rather than the status field.** Reaching
# MAGIC `detailed_state == ONLINE*` is necessary but NOT sufficient: there is a
# MAGIC window where the index reports ONLINE and still rejects searches, which
# MAGIC is why the smoke test below failed on essentially every first run. The
# MAGIC status field answers "has provisioning finished", which is not the
# MAGIC question we care about. Readiness here means "a search returns rows", so
# MAGIC that is what gets polled — the check and the thing it guards are now the
# MAGIC same operation, and the cell cannot pass while the next one would fail.

# COMMAND ----------

import time

_PROBE = "how do I avoid a Slack bot replying to its own messages"


def wait_until_queryable(idx, deadline_s=45 * 60, poll_s=15):
    """Block until a similarity_search succeeds. Returns the first result set."""
    started, last_state = time.time(), None
    while time.time() - started < deadline_s:
        try:
            state = idx.describe().get("status", {}).get("detailed_state", "")
        except Exception as e:                      # transient during creation
            state = f"describe failed: {type(e).__name__}"
        if state != last_state:
            print(f"  [{int(time.time() - started):>4}s] {state or 'unknown'}")
            last_state = state
        if str(state).startswith("ONLINE"):
            try:
                res = idx.similarity_search(
                    query_text=_PROBE,
                    columns=["id", "title", "text", "source"],
                    num_results=3,
                )
                print(f"Index queryable after {int(time.time() - started)}s")
                return res
            except Exception as e:
                # ONLINE but not yet serving — the exact gap this loop exists for
                print(f"  ONLINE but not serving yet ({type(e).__name__}); waiting")
        time.sleep(poll_s)
    raise TimeoutError(
        f"Index still not queryable after {deadline_s}s (last state: {last_state}). "
        f"Check the index in the Databricks UI before re-running."
    )


results = wait_until_queryable(index)

# COMMAND ----------

# MAGIC %md ## Smoke test
# MAGIC Uses the result the readiness poll already fetched, so this cannot fail
# MAGIC for timing reasons — if the cell above returned, the index answers.

# COMMAND ----------

rows = results["result"]["data_array"]
assert rows, "Index answered but returned no rows — the source table may be empty."
for r in rows:
    print(r[0], "-", r[1])
print(f"\n{len(rows)} results — knowledge base is live.")
