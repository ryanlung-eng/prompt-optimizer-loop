# Databricks notebook source
# MAGIC %md
# MAGIC # Re-validate Cached Conversations
# MAGIC
# MAGIC Re-runs ONLY the deterministic structural/schema validator (`validator.py` +
# MAGIC `check_params.js`) against conversations already saved in
# MAGIC `conversation_cache.json` — no KA calls, no judge calls, zero tokens spent.
# MAGIC Useful after changing the validator/schema (e.g. adding new node types) to
# MAGIC see how it now scores responses that were already generated, without
# MAGIC re-paying for a full eval run.
# MAGIC
# MAGIC This is a **read-only, separate notebook** — it never calls
# MAGIC `run_optimization_loop`, never touches n8n, and never writes back to
# MAGIC `config.yaml` or `conversation_cache.json`. It only reads the cache and
# MAGIC writes a fresh `revalidation.json` summary file.
# MAGIC
# MAGIC Only works if `_LOGIC_VERSION` (`evaluator.py` + `validator.py` +
# MAGIC `check_params.js`, hashed together) hasn't changed since the run you want
# MAGIC to re-check — if it has, the cache keys below won't match anything and
# MAGIC every input will show up as a miss. In that case there's no shortcut —
# MAGIC you need a real re-run via `workflow_builder_eval.py`.

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

import sys, os, asyncio, json
import nest_asyncio
nest_asyncio.apply()  # Databricks notebooks have their own event loop — this patches it

# Point to this repo if it's mounted as a Databricks Repo
sys.path.insert(0, "/Workspace/Users/ryan.lung@ibotta.com/prompt-optimizer-loop")

# COMMAND ----------

# MAGIC %md ## Configuration
# MAGIC No secrets needed — Databricks host + token are pulled automatically
# MAGIC from the cluster context.

# COMMAND ----------

# Pull host + token from cluster context — no secret needed
_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
os.environ["DATABRICKS_HOST"]  = "https://" + _ctx.browserHostName().get()
os.environ["DATABRICKS_TOKEN"] = _ctx.apiToken().get()

# COMMAND ----------

# MAGIC %md ## Load config + regenerate the same synthetic inputs
# MAGIC `generate_dataset` hits the on-disk dataset cache — no Haiku calls here,
# MAGIC just reconstructs the exact same 160 `SyntheticInput`s the original run used.

# COMMAND ----------

from pathlib import Path
from prompt_optimizer.config import load_config
from prompt_optimizer.evaluator import WorkflowEvaluator
from prompt_optimizer.synthetic_data import generate_dataset
from prompt_optimizer.validator import validate_workflow_json

CONFIG_PATH = "/Workspace/Users/ryan.lung@ibotta.com/prompt-optimizer-loop/config.yaml"  # matches workflow_builder_eval.py

config = load_config(CONFIG_PATH)
cache_dir = str(Path(config.synthetic_data.cache_path).parent)

inputs = asyncio.get_event_loop().run_until_complete(
    generate_dataset(config.synthetic_data, config.databricks)
)
evaluator = WorkflowEvaluator(config.databricks, cache_dir=cache_dir)

endpoint_url = f"{config.databricks.workspace_url}/serving-endpoints/{config.databricks.eval_endpoint}/invocations"
system_prompt = config.prompts["Workflow Builder"]

print(f"{len(inputs)} synthetic inputs loaded, {len(evaluator._cache)} entries in conversation cache")

# COMMAND ----------

# MAGIC %md ## Look up each input's cached response, re-validate with the current schema checker

# COMMAND ----------

hits, misses = 0, 0
never_json, invalid, valid = [], [], []

for inp in inputs:
    key = evaluator._cache_key(system_prompt, inp, endpoint_url)
    cached = evaluator._cache.get(key)
    if cached is None:
        misses += 1
        continue
    hits += 1
    response = cached["response"]
    structural = validate_workflow_json(response)
    record = {"category": inp.category, "input": inp.text[:80], "errors": structural.errors}
    if not structural.is_json:
        never_json.append(record)
    elif not structural.valid:
        invalid.append(record)
    else:
        valid.append(record)

print(f"Cache: {hits} hits, {misses} misses (misses need a fresh eval run to re-check)")
print(f"Never produced JSON: {len(never_json)}/{hits}")
print(f"Attempted, still invalid: {len(invalid)}/{hits}")
print(f"Structurally valid: {len(valid)}/{hits}")

if invalid:
    print("\n--- Invalid: unique structural errors ---")
    seen = []
    for r in invalid:
        for e in r["errors"]:
            if e not in seen:
                seen.append(e)
                print(f"  • {e}")

# COMMAND ----------

# MAGIC %md ## Write full detail for further digging

# COMMAND ----------

out = Path("/Workspace/Users/ryan.lung@ibotta.com/n8n-optimizer-cache/revalidation.json")
out.write_text(json.dumps({"never_json": never_json, "invalid": invalid, "valid": valid}, indent=2))
print(f"Full detail written to {out}")
