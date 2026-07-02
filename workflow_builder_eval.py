# Databricks notebook source
# MAGIC %md
# MAGIC # Workflow Builder Prompt Optimizer
# MAGIC
# MAGIC Runs the prompt optimization loop against the Workflow Builder KA node.
# MAGIC Each iteration: generate synthetic inputs → evaluate against raw LLM → judge → improve prompt.
# MAGIC
# MAGIC n8n staging is unreachable from this network (firewall blocks the Databricks
# MAGIC VPC), so this notebook never reads or writes n8n. Paste the current prompt
# MAGIC into config.yaml's `prompts` section — the best-scoring prompt is printed in
# MAGIC full at the end of the run for you to copy back into n8n yourself.
# MAGIC
# MAGIC **Widgets:**
# MAGIC - `mode`: `evaluate` (score only) | `optimize` (score + improve) | `generate` (synthetic data only)
# MAGIC - `max_iterations`: override config default

# COMMAND ----------

# MAGIC %pip install httpx tenacity rich nest_asyncio pyyaml mlflow -q

# COMMAND ----------

import sys, os, asyncio
import nest_asyncio
nest_asyncio.apply()  # Databricks notebooks have their own event loop — this patches it

# Point to this repo if it's mounted as a Databricks Repo
sys.path.insert(0, "/Workspace/Users/ryan.lung@ibotta.com/prompt-optimizer-loop")

# COMMAND ----------

# MAGIC %md ## Configuration
# MAGIC No secrets needed — Databricks host + token are pulled automatically
# MAGIC from the cluster context.

# COMMAND ----------

dbutils.widgets.dropdown("mode", "evaluate", ["evaluate", "optimize", "generate"], "Mode")
dbutils.widgets.text("max_iterations", "", "Max iterations (blank = use config)")

# COMMAND ----------

# Pull host + token from cluster context — no secret needed
_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
os.environ["DATABRICKS_HOST"]  = "https://" + _ctx.browserHostName().get()
os.environ["DATABRICKS_TOKEN"] = _ctx.apiToken().get()

# COMMAND ----------

# MAGIC %md ## Run

# COMMAND ----------

from prompt_optimizer.config import load_config
from prompt_optimizer.loop import run_optimization_loop

cfg = load_config("/Workspace/Users/ryan.lung@ibotta.com/prompt-optimizer-loop/config.yaml")

# Widget overrides
mode = dbutils.widgets.get("mode")
max_iter_str = dbutils.widgets.get("max_iterations").strip()
if max_iter_str:
    cfg.optimizer.max_iterations = int(max_iter_str)

asyncio.get_event_loop().run_until_complete(run_optimization_loop(
    config=cfg,
    generate_only=(mode == "generate"),
    evaluate_only=(mode == "evaluate"),
))
