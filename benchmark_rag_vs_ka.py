# Databricks notebook source
# MAGIC %md
# MAGIC # Hard-Scenario Benchmark: Production KA vs. Custom RAG Pipeline
# MAGIC
# MAGIC Compares the production Agent Bricks Knowledge Assistant endpoint against
# MAGIC the new custom RAG pipeline (frozen `instructions.md` + retrieval over
# MAGIC `knowledge-base-upload/` via `rag_setup.py`'s index) on the Layer 4
# MAGIC hand-crafted hard scenarios (`hard_scenarios.py`) — 18 deliberately
# MAGIC difficult multi-step scenarios (self-triggering loops, unwired AI
# MAGIC sub-nodes, multi-hop chains), rather than the easier 200-item synthetic
# MAGIC trigger×output set.
# MAGIC
# MAGIC **Prerequisite: run `rag_setup.py` first** — this notebook queries the
# MAGIC index it creates; if that index doesn't exist yet, the custom_rag arm
# MAGIC will fail on every scenario.

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

# MAGIC %md ## Run

# COMMAND ----------

from prompt_optimizer.config import load_config
from prompt_optimizer import benchmark

cfg = load_config("/Workspace/Users/ryan.lung@ibotta.com/prompt-optimizer-loop/config.yaml")

results = asyncio.get_event_loop().run_until_complete(benchmark.run_hard(cfg))

# COMMAND ----------

# MAGIC %md ## Inspect every custom_rag failure in detail

# COMMAND ----------

for r in results["custom_rag"]:
    if not r.structural.valid or r.structural.warnings or r.soundness_issues:
        print("Scenario:", r.input.category)
        print("Structural errors:", r.structural.errors)
        print("Warnings (Layer 2):", r.structural.warnings)
        print("Soundness issues (Layer 3):", r.soundness_issues)
        print("Actual response:", r.actual_response[:2000])
        print("---")
