# Databricks notebook source
# MAGIC %md
# MAGIC # Refresh the knowledge base, then deploy the workflow-builder endpoint
# MAGIC
# MAGIC One notebook to run after uploading new/edited docs to the volume. It
# MAGIC runs the two existing notebooks in order and nothing else — the real
# MAGIC logic stays in them, so this cannot drift away from what they do.
# MAGIC
# MAGIC 1. `rag_setup.py` — re-chunks `docs/` + `examples/` from the volume,
# MAGIC    overwrites the Delta table, recreates the vector index (a full
# MAGIC    re-embed), and waits until the index actually answers queries.
# MAGIC 2. `deploy_workflow_builder_endpoint.py` — logs the pipeline as an
# MAGIC    MLflow model and creates/updates the `n8n-workflow-builder-rag`
# MAGIC    serving endpoint.
# MAGIC
# MAGIC **Why `dbutils.notebook.run` and not `%run`.** Both notebooks call
# MAGIC `dbutils.library.restartPython()` after their `%pip install`. A restart
# MAGIC inside a `%run` would tear down this notebook's own state mid-flight.
# MAGIC `dbutils.notebook.run` gives each one its own Python context, so the
# MAGIC restarts stay contained.
# MAGIC
# MAGIC **Order matters.** The deploy notebook smoke-tests the pipeline in
# MAGIC process, which queries the index — so the KB has to be live first.
# MAGIC
# MAGIC **Upload first.** This reads the volume; it does not put anything there.
# MAGIC Edited docs must already be in
# MAGIC `/Volumes/dev/platform/automation-builder/docs` (or `examples`) before
# MAGIC you run this, or it will faithfully re-embed the old contents.
# MAGIC
# MAGIC Safe to re-run. Both steps are idempotent: the index is recreated from
# MAGIC scratch and the endpoint is updated in place if it already exists.

# COMMAND ----------

# MAGIC %md ## Settings

# COMMAND ----------

# Generous. A full re-embed of the corpus plus endpoint provisioning is a
# several-minute operation on a good day, and a timeout that fires early leaves
# a half-built index behind — worse than waiting.
KB_TIMEOUT_S = 60 * 60
DEPLOY_TIMEOUT_S = 60 * 60

# Set False to deploy without touching the knowledge base (e.g. you only
# changed config.yaml or pipeline code, not the docs).
#
# Currently False: the endpoint needs redeploying for the scope mode, and that
# is a pure code + config change — the index content is unaffected, so a full
# re-chunk and re-embed would be several minutes of work for no difference.
#
# SET THIS BACK TO True after uploading edited docs to the volume. Leaving it
# False is the quiet failure mode here: the deploy succeeds, the endpoint looks
# healthy, and it keeps serving the old corpus.
REFRESH_KB = False

# Set False to refresh the knowledge base only, without redeploying.
# Worth knowing: the endpoint queries the index at REQUEST time, so a pure KB
# refresh needs no redeploy at all. A redeploy is only needed when config.yaml
# or the pipeline code changed, because those are baked into the model version.
DEPLOY_ENDPOINT = True

# COMMAND ----------

# MAGIC %md ## Run

# COMMAND ----------

import time


def _step(name, path, timeout_s):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    started = time.time()
    try:
        out = dbutils.notebook.run(path, timeout_s)
    except Exception as e:
        # Say which step died and how far it got. A bare traceback here is
        # ambiguous about whether the KB or the deploy failed, and the two have
        # completely different recovery paths.
        raise RuntimeError(
            f"{name} FAILED after {int(time.time() - started)}s. Open "
            f"{path} and re-run it directly to see the failing cell. "
            f"Underlying error: {e}"
        ) from e
    print(f"\n{name} finished in {int(time.time() - started)}s.")
    return out


if REFRESH_KB:
    _step("1/2  Knowledge base refresh", "./rag_setup", KB_TIMEOUT_S)
else:
    print("1/2  Knowledge base refresh SKIPPED (REFRESH_KB=False)")

if DEPLOY_ENDPOINT:
    _step("2/2  Endpoint deploy", "./deploy_workflow_builder_endpoint",
          DEPLOY_TIMEOUT_S)
else:
    print("2/2  Endpoint deploy SKIPPED (DEPLOY_ENDPOINT=False)")

print("\nDone.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Then, in n8n
# MAGIC
# MAGIC `automation-builder-new` (wYgXYsfeAuJSSrR7) is already wired to this
# MAGIC endpoint via the `Workflow Builder` node (`CUSTOM.databricks`, credential
# MAGIC `SP Prod - Priority Operations`).
# MAGIC
# MAGIC Run it from Slack and check BOTH paths before cutting over:
# MAGIC
# MAGIC - a buildable request → `status=workflow`, file posted by
# MAGIC   `Send Download Link`
# MAGIC - a vague request → `status=question`, relayed by `Send a message12`
# MAGIC
# MAGIC Only then unpublish the old `automation-builder`.
# MAGIC
# MAGIC Worth watching on the first real runs: `repair_rounds` in the response.
# MAGIC Consistently hitting the cap (3) means the repair loop is churning rather
# MAGIC than converging — visible here as a number rather than as mystery latency.
