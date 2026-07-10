# Databricks notebook source
# MAGIC %md
# MAGIC # Workflow Builder Benchmark: value of prompt engineering + knowledge base
# MAGIC
# MAGIC Separate from the eval/optimize loop (`workflow_builder_eval.py`) — this is an
# MAGIC occasional-use comparison, not something you'd run every time you tweak the
# MAGIC prompt. Runs the SAME synthetic inputs, judge, and structural validator across
# MAGIC three arms that all get the IDENTICAL system prompt, varying only which
# MAGIC endpoint answers and whether a knowledge base is available to it:
# MAGIC
# MAGIC - **no_knowledge** — raw Sonnet (`generation_endpoint`), no KB access at all.
# MAGIC - **knowledge_injected** — raw Sonnet, the full flattened KB corpus
# MAGIC   (`knowledge-base-upload/`, ~117k tokens) pasted directly into the prompt.
# MAGIC - **production** — the actual KA endpoint (`eval_endpoint`), whatever its own
# MAGIC   internal knowledge access does.
# MAGIC
# MAGIC Neither raw-Sonnet arm has web search — this all runs through Databricks-hosted
# MAGIC endpoints, not the real claude.ai/API, so that's not a confound either way.

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

# MAGIC %pip install httpx tenacity rich nest_asyncio pyyaml mlflow -q
# MAGIC %pip install --upgrade typing_extensions -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import sys, os, asyncio
import nest_asyncio
nest_asyncio.apply()  # Databricks notebooks have their own event loop — this patches it

# Point to this repo if it's mounted as a Databricks Repo
sys.path.insert(0, "/Workspace/Users/ryan.lung@ibotta.com/prompt-optimizer-loop")

# COMMAND ----------

# Pull host + token from cluster context — no secret needed
_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
os.environ["DATABRICKS_HOST"]  = "https://" + _ctx.browserHostName().get()
os.environ["DATABRICKS_TOKEN"] = _ctx.apiToken().get()

# COMMAND ----------

# MAGIC %md ## Run

# COMMAND ----------

from pathlib import Path

cache_path = Path("/Workspace/Users/ryan.lung@ibotta.com/n8n-optimizer-cache/conversation_cache.json")

if cache_path.exists():
    import json
    entry_count = len(json.loads(cache_path.read_text()))
    cache_path.unlink()
    print(f"Deleted {cache_path} ({entry_count} cached conversations).")
else:
    print(f"No cache file found at {cache_path} — nothing to clear.")

# COMMAND ----------

from prompt_optimizer.config import load_config
from prompt_optimizer import benchmark

cfg = load_config("/Workspace/Users/ryan.lung@ibotta.com/prompt-optimizer-loop/config.yaml")

results = asyncio.get_event_loop().run_until_complete(benchmark.run(cfg))

# COMMAND ----------

for r in results["production"]:
    if not r.structural.valid:
        print("Request:", r.input.text)
        print("Errors:", r.structural.errors)
        print("Actual response:", r.actual_response[:3000])
