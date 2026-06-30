# Databricks notebook source
# MAGIC %md
# MAGIC # Workflow Builder Prompt Optimizer
# MAGIC
# MAGIC Runs the prompt optimization loop against the Workflow Builder KA node.
# MAGIC Each iteration: generate synthetic inputs → evaluate against raw LLM → judge → improve prompt → write back to n8n.
# MAGIC
# MAGIC **Widgets:**
# MAGIC - `mode`: `evaluate` (score only) | `optimize` (score + improve) | `generate` (synthetic data only)
# MAGIC - `max_iterations`: override config default
# MAGIC - `dry_run`: if checked, never writes back to n8n

# COMMAND ----------

# MAGIC %pip install anthropic httpx tenacity rich nest_asyncio -q

# COMMAND ----------

import sys, os, asyncio
import nest_asyncio
nest_asyncio.apply()  # Databricks notebooks have their own event loop — this patches it

# Point to this repo if it's mounted as a Databricks Repo
sys.path.insert(0, "/Workspace/Repos/ryan.lung@ibotta.com/n8n")

# COMMAND ----------

# MAGIC %md ## Configuration
# MAGIC Only one secret needed — the n8n API key.
# MAGIC Databricks host + token are pulled automatically from the cluster context.
# MAGIC ```
# MAGIC databricks secrets create-scope n8n-optimizer
# MAGIC databricks secrets put-secret n8n-optimizer N8N_API_KEY --string-value <key>
# MAGIC ```

# COMMAND ----------

dbutils.widgets.dropdown("mode", "evaluate", ["evaluate", "optimize", "generate"], "Mode")
dbutils.widgets.text("max_iterations", "", "Max iterations (blank = use config)")
dbutils.widgets.dropdown("dry_run", "False", ["True", "False"], "Dry run (no n8n writes)")

# COMMAND ----------

SECRET_SCOPE = "n8n-optimizer"

# Pull host + token from cluster context — no secret needed
_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
os.environ["DATABRICKS_HOST"]  = "https://" + _ctx.browserHostName().get()
os.environ["DATABRICKS_TOKEN"] = _ctx.apiToken().get()
os.environ["N8N_BASE_URL"]     = "https://n8n.ops.int.staging.ibops.net"
os.environ["N8N_API_KEY"]      = dbutils.secrets.get(SECRET_SCOPE, "N8N_API_KEY")
os.environ["N8N_WORKFLOW_ID"]   = "8hy6AFy4xdwqLNA4"

# COMMAND ----------

# MAGIC %md ## Run

# COMMAND ----------

from prompt_optimizer.config import load_config
from prompt_optimizer.loop import run_optimization_loop

cfg = load_config("/Workspace/Repos/ryan.lung@ibotta.com/n8n/config.yaml")

# Widget overrides
mode = dbutils.widgets.get("mode")
max_iter_str = dbutils.widgets.get("max_iterations").strip()
if max_iter_str:
    cfg.optimizer.max_iterations = int(max_iter_str)

dry_run = dbutils.widgets.get("dry_run") == "true"

asyncio.get_event_loop().run_until_complete(run_optimization_loop(
    config=cfg,
    dry_run=dry_run,
    generate_only=(mode == "generate"),
    evaluate_only=(mode == "evaluate"),
))
