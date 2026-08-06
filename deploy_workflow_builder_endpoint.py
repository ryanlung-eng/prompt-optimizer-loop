# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy the custom-RAG workflow builder as one serving endpoint
# MAGIC
# MAGIC Packages retrieval + relevance filter + generation + validation + the
# MAGIC self-repair loop behind a single Databricks Model Serving endpoint, so
# MAGIC n8n makes ONE call and gets back a finished workflow or a clarifying
# MAGIC question.
# MAGIC
# MAGIC **What this replaces in n8n.** The production automation-builder ran
# MAGIC Workflow Builder → Workflow Validator → retry loop → Workflow Checker →
# MAGIC Validator Parser → Workflow Fixer as separate nodes. All of that happens
# MAGIC inside `build()` now. `automation-builder-new` (wYgXYsfeAuJSSrR7) is
# MAGIC already rewired for it.
# MAGIC
# MAGIC **This is NOT a Knowledge Assistant endpoint,** so n8n cannot call it
# MAGIC with the CUSTOM.ibottaKnowledgeAssistant node — that node speaks a
# MAGIC KA-specific wire format, while this is an MLflow PyFunc taking
# MAGIC `{inputs: [...]}` and returning `{predictions: [...]}`. The workflow uses
# MAGIC the `CUSTOM.databricks` node (resource: modelServing, operation:
# MAGIC queryEndpoint) — NOT the identically-named `n8n-nodes-base.databricks`,
# MAGIC which wants a `databricksApi` credential this instance does not have;
# MAGIC CUSTOM.databricks takes the `databricks` type, which it does. An Unwrap
# MAGIC Builder Response Code node then flattens `predictions[0]`, so everything
# MAGIC downstream of it is unchanged.
# MAGIC
# MAGIC **Why the schema check is pure Python now.** The old validator shelled
# MAGIC out to check_params.js, which would have meant a Node runtime plus
# MAGIC node_modules inside this image and a process spawn per request.
# MAGIC `schema_check.py` is asserted equivalent to the JS by
# MAGIC `n8n_schema_check/equivalence_check.py` — run it if you touch either.
# MAGIC
# MAGIC **The one packaging catch:** the checker needs n8n's node manifest
# MAGIC (`nodes.json`, ~8 MB) at runtime, and that file is not in git — it lives
# MAGIC inside the gitignored `node_modules/`. The staging cell below sources it
# MAGIC (copying an existing one, or running `npm install` to fetch it) and puts
# MAGIC it beside the module so it ships inside the code artifact.

# COMMAND ----------

# MAGIC %md ## Pull the latest commit into this workspace

# COMMAND ----------

_REPO_PATH = "/Workspace/Users/ryan.lung@ibotta.com/prompt-optimizer-loop"

try:
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    _repo = next((r for r in w.repos.list() if r.path == _REPO_PATH), None)
    if _repo is None:
        print(f"No Databricks Repo at {_REPO_PATH} — pull/sync manually before continuing.")
    else:
        w.repos.update(repo_id=_repo.id, branch="main")
        print(f"Updated {_REPO_PATH} to latest main.")
except Exception as e:
    print(f"Could not auto-pull — pull manually via the Repos UI. Error: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch the node manifest
# MAGIC Same recipe `workflow_builder_eval.py` has always used: the cluster has no
# MAGIC npm, so download a portable Node and use its npm. Lands in
# MAGIC `/tmp/n8n_schema_check_cache`, which is exactly where `validator.py`'s
# MAGIC `_LOCAL_CACHE_DIR` already looks — so this also makes the checker work in
# MAGIC this notebook, not only inside the model that gets logged.
# MAGIC
# MAGIC Skipped automatically if the manifest is already there, so a re-run on a
# MAGIC warm cluster costs nothing.

# COMMAND ----------

# MAGIC %sh
# MAGIC set -e
# MAGIC MANIFEST=/tmp/n8n_schema_check_cache/node_modules/n8n-nodes-base/dist/types/nodes.json
# MAGIC if [ -f "$MANIFEST" ]; then
# MAGIC   echo "Manifest already cached: $(du -h "$MANIFEST" | cut -f1)"
# MAGIC   exit 0
# MAGIC fi
# MAGIC
# MAGIC if [ ! -x /tmp/node22/bin/npm ]; then
# MAGIC   mkdir -p /tmp/node22 && cd /tmp/node22
# MAGIC   curl -fsSL -o node.tar.gz https://nodejs.org/dist/v22.9.0/node-v22.9.0-linux-x64.tar.gz
# MAGIC   tar -xzf node.tar.gz --strip-components=1
# MAGIC fi
# MAGIC /tmp/node22/bin/npm --version
# MAGIC
# MAGIC REPO=/Workspace/Users/ryan.lung@ibotta.com/prompt-optimizer-loop
# MAGIC mkdir -p /tmp/n8n_schema_check_cache
# MAGIC cp $REPO/prompt_optimizer/n8n_schema_check/check_params.js /tmp/n8n_schema_check_cache/
# MAGIC cp $REPO/prompt_optimizer/n8n_schema_check/package.json    /tmp/n8n_schema_check_cache/
# MAGIC cd /tmp/n8n_schema_check_cache && /tmp/node22/bin/npm install --ignore-scripts
# MAGIC ls -la "$MANIFEST"

# COMMAND ----------

# MAGIC %pip install mlflow httpx tenacity pyyaml databricks-ai-search databricks-sdk -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/Workspace/Users/ryan.lung@ibotta.com/prompt-optimizer-loop")

_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
os.environ["DATABRICKS_HOST"] = "https://" + _ctx.browserHostName().get()
os.environ["DATABRICKS_TOKEN"] = _ctx.apiToken().get()

REPO = Path("/Workspace/Users/ryan.lung@ibotta.com/prompt-optimizer-loop")
ENDPOINT_NAME = "n8n-workflow-builder-rag"   # must match servingEndpointId in n8n
MODEL_NAME = "dev.platform.n8n_workflow_builder"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage the package and materialise the node manifest
# MAGIC
# MAGIC `schema_check.py` needs n8n's `nodes.json` (~8 MB) at runtime. It normally
# MAGIC reads it out of `node_modules`, which will not exist in the serving image,
# MAGIC so a copy has to ship inside the code artifact.
# MAGIC
# MAGIC **`nodes.json` is not in git and cannot be.** It lives inside
# MAGIC `node_modules/`, which is gitignored, and it is ~8 MB of a third-party
# MAGIC package — not something to vendor into a public repo. So a fresh clone
# MAGIC never has it, and this notebook has to PRODUCE it rather than assume it.
# MAGIC
# MAGIC The package is staged to scratch disk first, so the manifest can be
# MAGIC dropped beside the checker without writing into the Repo folder (read-only
# MAGIC on some workspaces, and a source of confusing diffs even when it isn't).
# MAGIC That staged copy is then prepended to `sys.path` and is what gets logged —
# MAGIC so the smoke tests below exercise exactly the tree that ships.
# MAGIC
# MAGIC The manifest itself comes from the `%sh` cell near the top, which uses
# MAGIC the same portable-Node recipe `workflow_builder_eval.py` has always used.

# COMMAND ----------

import json
import tempfile

# Stage the package to /tmp so the manifest can sit beside the checker without
# writing into the Repo folder — node_modules-shaped writes under /Workspace
# break Databricks Repos' git-status UI, which is why validator.py keeps its
# cache out here too.
STAGE = Path(tempfile.gettempdir()) / "wf_builder_pkg"
PKG = STAGE / "prompt_optimizer"

if STAGE.exists():
    shutil.rmtree(STAGE)
STAGE.mkdir(parents=True)
shutil.copytree(
    REPO / "prompt_optimizer", PKG,
    ignore=shutil.ignore_patterns("node_modules", "__pycache__", "*.pyc"),
)
print(f"Staged package -> {PKG}")

# The %sh cell above put the manifest in validator.py's own _LOCAL_CACHE_DIR.
# Look there first, then at anything already sitting in the repo checkout.
_dst = PKG / "n8n_schema_check/nodes.json"
_nm = Path("node_modules/n8n-nodes-base/dist/types/nodes.json")
_candidates = [
    Path(tempfile.gettempdir()) / "n8n_schema_check_cache" / _nm,
    REPO / "prompt_optimizer/n8n_schema_check/nodes.json",
    REPO / "prompt_optimizer/n8n_schema_check" / _nm,
]
_found = next((p for p in _candidates if p.exists()), None)
if _found is None:
    raise FileNotFoundError(
        "nodes.json not found in any of:\n  "
        + "\n  ".join(str(c) for c in _candidates)
        + "\n\nThe %sh cell above should have produced the first one. Re-run it "
          "and check its output — without the manifest the endpoint would ship a "
          "schema check that silently finds nothing."
    )

shutil.copyfile(_found, _dst)
print(f"Manifest from {_found}")
print(f"Bundled manifest: {_dst} ({_dst.stat().st_size:,} bytes)")
assert _dst.stat().st_size > 1_000_000, "manifest implausibly small — truncated copy?"

# From here on, import from the STAGED copy, so everything verified below is
# the same tree that gets logged.
sys.path.insert(0, str(STAGE))

# COMMAND ----------

# MAGIC %md
# MAGIC Prove the checker actually works from the bundled copy BEFORE logging a
# MAGIC model that depends on it. A silently-degraded validator — one that
# MAGIC returns "no issues" because the manifest failed to load — is the failure
# MAGIC mode worth spending a cell to rule out, since it looks like success.

# COMMAND ----------

from prompt_optimizer.schema_check import check_workflow

_loaded_from = Path(sys.modules["prompt_optimizer.schema_check"].__file__)
assert str(_loaded_from).startswith(str(STAGE)), (
    f"imported the Repo copy ({_loaded_from}), not the staged one — sys.path "
    f"ordering is wrong, so this probe would not be testing what ships."
)

_probe = check_workflow({"nodes": [{"name": "S", "type": "n8n-nodes-base.slack",
                                    "typeVersion": 2.3,
                                    "parameters": {"resource": "message",
                                                   "operation": "post",
                                                   "totallyInvented": "x"}}]})
assert _probe["issues"], f"schema check returned nothing — manifest not loading: {_probe}"
assert "totallyInvented" in _probe["issues"][0]["unknownParams"], _probe
print("Schema check live, caught:", _probe["issues"][0]["unknownParams"])

# COMMAND ----------

# MAGIC %md ## Smoke-test the pipeline in-process before deploying it

# COMMAND ----------

from prompt_optimizer.config import load_config
from prompt_optimizer.serving import WorkflowBuilderPipeline

cfg = load_config(str(REPO / "config.yaml"))
pipeline = WorkflowBuilderPipeline(cfg)

_REQUEST = ("When a new row is added to my Google Sheet, post a message to "
            "#general with the row contents.")

# Scope mode first — it is cheaper, and if retrieval or the endpoint is broken
# it fails here in seconds rather than after a full build with repair rounds.
_scoped = pipeline.scope(_REQUEST)
print("scope status :", _scoped["status"])
print("scope reply  :", (_scoped["data"] or "")[:300])
assert _scoped["status"] in ("question", "error"), _scoped
assert not (_scoped["data"] or "").lstrip().startswith("{"), \
    "scope mode emitted JSON — the no-workflow-JSON guardrail is not holding"

_result = pipeline.build(_REQUEST)
print("status      :", _result["status"])
print("valid       :", _result["valid"])
print("repair_rounds:", _result["repair_rounds"])
print("errors      :", _result["errors"][:2])
print("kept_sources:", _result["kept_sources"][:5])
assert _result["status"] in ("workflow", "question"), _result

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the model
# MAGIC config.yaml is logged as an ARTIFACT so the prompt text and retrieval
# MAGIC settings are pinned to this model version — otherwise the deployed
# MAGIC system could drift away from the configuration that was benchmarked.

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature

from prompt_optimizer.serving_model import WorkflowBuilderModel

mlflow.set_registry_uri("databricks-uc")

# All four fields must appear in the signature, not just `question`: Model
# Serving validates the request against it, so a field missing here is a field
# n8n cannot send. credentials/user_id/minutes_saved are supplied per request
# by the workflow (Credential Parser, Slack Trigger, AI Agent) rather than
# hardcoded in config.yaml.
_input_example = [{
    "question": "Post a Slack message to #general every Monday at 9am.",
    "credentials": 'The user has the following credentials configured:\n'
                   'Slack enabled, id: "EXAMPLE-slack-0001"',
    "user_id": "U000EXAMPLE",
    "minutes_saved": "30",
    # "" or "build" -> workflow JSON; "scope" -> conversational scoping.
    # Must appear in the signature or n8n cannot send it and every request
    # silently falls back to build.
    "mode": "build",
}]
_output_example = [{"status": "workflow", "workflow": {}, "message": "", "valid": True,
                    "errors": [], "warnings": [], "repair_rounds": 0,
                    "kept_sources": [], "data": "{}"}]

# Declaring the Databricks resources this model calls lets Model Serving do
# automatic authentication passthrough: it mints a short-lived credential for
# exactly these resources and injects it into the container. That is why no
# secret scope and no long-lived PAT are needed below.
#
# The notebooks get their token from `dbutils ... apiToken()`, which is the
# running user's identity. A serving replica has no notebook context and no
# user, so that call does not exist there — which is the whole reason this
# needs saying explicitly rather than inheriting what the notebooks do.
from mlflow.models.resources import (
    DatabricksServingEndpoint, DatabricksVectorSearchIndex,
)

_resources = [
    DatabricksVectorSearchIndex(index_name=cfg.rag.index_name),
    DatabricksServingEndpoint(endpoint_name=cfg.rag.generation_endpoint),
    DatabricksServingEndpoint(endpoint_name=cfg.databricks.fast_generation_endpoint),
]
print("declared resources:", [r.to_dict() for r in _resources])

with mlflow.start_run(run_name="n8n-workflow-builder-rag") as run:
    info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=WorkflowBuilderModel(),
        artifacts={"config": str(REPO / "config.yaml")},
        # The STAGED package, not the Repo one — only the staged copy has
        # nodes.json beside the checker, and shipping the Repo copy would give
        # an endpoint whose schema check silently finds nothing.
        code_paths=[str(PKG)],
        signature=infer_signature(_input_example, _output_example),
        input_example=_input_example,
        registered_model_name=MODEL_NAME,
        resources=_resources,
        pip_requirements=["mlflow", "httpx", "tenacity", "pyyaml",
                          "databricks-ai-search", "pandas"],
    )
print("logged:", info.model_uri)

# Unity Catalog does not populate RegisteredModel.latest_versions (it comes back
# None, hence the TypeError this replaces). Take the version THIS run registered
# straight off the ModelInfo — which is also more correct than asking for the
# latest, since "latest" would silently deploy someone else's concurrent
# registration rather than the model just smoke-tested above.
_version = getattr(info, "registered_model_version", None)

if _version is None:                       # older mlflow: ask, newest first
    _versions = mlflow.MlflowClient().search_model_versions(
        f"name='{MODEL_NAME}'", order_by=["version_number DESC"], max_results=1,
    )
    if not _versions:
        raise RuntimeError(
            f"{MODEL_NAME} has no registered versions — did log_model's "
            f"registered_model_name take effect?"
        )
    _version = _versions[0].version
    print("(fell back to search_model_versions)")

print("registered version:", _version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create / update the serving endpoint
# MAGIC The endpoint calls Vector Search and two model-serving endpoints at run
# MAGIC time, so it needs credentials of its own — a serving replica has no
# MAGIC notebook context and therefore no `dbutils ... apiToken()` to borrow the
# MAGIC way every notebook in this repo does.
# MAGIC
# MAGIC Those credentials come from the `resources=` declared at log time
# MAGIC (automatic authentication passthrough), so there is **no secret scope to
# MAGIC create and no long-lived PAT to rotate**. If passthrough ever fails —
# MAGIC for a resource type it does not cover — the commented fallback below
# MAGIC supplies a secret-backed token instead.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput, ServedEntityInput,
)

w = WorkspaceClient()

# Fallback only. Uncomment and point at a real scope/key ONLY if passthrough
# turns out not to cover something; then pass environment_vars=_env_vars below.
# The host may be a literal; the token must never be.
# _env_vars = {
#     "DATABRICKS_HOST": os.environ["DATABRICKS_HOST"],
#     "DATABRICKS_TOKEN": "{{secrets/<scope>/<key>}}",
# }

_entity = ServedEntityInput(
    entity_name=MODEL_NAME,
    entity_version=_version,
    workload_size="Small",
    scale_to_zero_enabled=True,
)

def _wait_until_updatable(name: str, timeout_s: int = 2400, poll_s: int = 20) -> None:
    """Block while a config update is in flight.

    Model Serving rejects a new update with ResourceConflict while one is in
    progress, and a FAILED deployment keeps retrying for a while before it
    settles — so the window right after a failure is exactly when you want to
    push a fix and exactly when it is refused. Waiting here beats making the
    operator guess how long to sit on their hands.

    UPDATE_FAILED is a terminal state, not an in-flight one: it means the last
    attempt finished and lost. That is updatable, and is the normal case for a
    redeploy-after-failure, so it must not be treated as "still going".
    """
    import time as _time

    started, last = _time.time(), None
    while _time.time() - started < timeout_s:
        state = w.serving_endpoints.get(name=name).state
        update = str(getattr(state, "config_update", "") or "")
        ready = str(getattr(state, "ready", "") or "")
        if update != last:
            print(f"  [{int(_time.time() - started):>4}s] config_update={update or 'unknown'} ready={ready or 'unknown'}")
            last = update
        if "IN_PROGRESS" not in update.upper():
            return
        _time.sleep(poll_s)
    raise TimeoutError(
        f"{name} still updating after {timeout_s}s (last: {last}). Check the "
        f"endpoint's Logs tab — a container that cannot start will retry for a "
        f"long time before the update is marked failed."
    )


_existing = next((e for e in w.serving_endpoints.list() if e.name == ENDPOINT_NAME), None)
if _existing is not None:
    print(f"{ENDPOINT_NAME} exists — waiting for any in-flight update to settle…")
    _wait_until_updatable(ENDPOINT_NAME)

if _existing is None:
    w.serving_endpoints.create(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(served_entities=[_entity]),
    )
    print(f"Creating endpoint {ENDPOINT_NAME} — this takes several minutes.")
else:
    w.serving_endpoints.update_config(name=ENDPOINT_NAME, served_entities=[_entity])
    print(f"Updating endpoint {ENDPOINT_NAME} to version {_version}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grant the n8n service principal permission to query
# MAGIC
# MAGIC **This is what causes `API Error: 403 Forbidden` in n8n.** The endpoint
# MAGIC is created by whoever runs this notebook, and only that identity can
# MAGIC query it. n8n calls it with the `SP Prod - Priority Operations`
# MAGIC credential — a different principal, which has no permission by default.
# MAGIC
# MAGIC 403 is worth distinguishing from the failures it gets mistaken for: a
# MAGIC still-provisioning endpoint returns 404, and a wrong/missing credential
# MAGIC returns 401. A 403 means the call authenticated fine and the principal
# MAGIC simply is not allowed — waiting will never fix it.
# MAGIC
# MAGIC Set the service principal's application ID below. Find it in the
# MAGIC Databricks admin console under Identity and access → Service principals,
# MAGIC matching whichever principal issued the token in that n8n credential.

# COMMAND ----------

N8N_SERVICE_PRINCIPAL = None   # e.g. "1234abcd-56ef-78ab-90cd-1234567890ab"

if not N8N_SERVICE_PRINCIPAL:
    print("N8N_SERVICE_PRINCIPAL not set — SKIPPING the grant.\n"
          "n8n will get 403 Forbidden from this endpoint until it is granted "
          "CAN_QUERY, either by setting this and re-running the cell, or via "
          "Serving > the endpoint > Permissions in the UI.")
else:
    from databricks.sdk.service.serving import (
        ServingEndpointAccessControlRequest, ServingEndpointPermissionLevel,
    )

    _ep = w.serving_endpoints.get(name=ENDPOINT_NAME)
    # update_permissions, not set_permissions: set_ REPLACES the whole ACL and
    # would silently strip everyone else's access, including your own.
    w.serving_endpoints.update_permissions(
        serving_endpoint_id=_ep.id,
        access_control_list=[
            ServingEndpointAccessControlRequest(
                service_principal_name=N8N_SERVICE_PRINCIPAL,
                permission_level=ServingEndpointPermissionLevel.CAN_QUERY,
            )
        ],
    )
    print(f"Granted CAN_QUERY on {ENDPOINT_NAME} to {N8N_SERVICE_PRINCIPAL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke-test the deployed endpoint
# MAGIC Run once the endpoint reports READY. This is the same call n8n's
# MAGIC Databricks node (resource: modelServing, operation: queryEndpoint) makes.

# COMMAND ----------

import json

import httpx

_url = f"{os.environ['DATABRICKS_HOST']}/serving-endpoints/{ENDPOINT_NAME}/invocations"
_resp = httpx.post(
    _url,
    headers={"Authorization": f"Bearer {os.environ['DATABRICKS_TOKEN']}",
             "Content-Type": "application/json"},
    json={"inputs": [{
        "question": "Every Monday at 9am, post a summary to #general.",
        "credentials": 'The user has the following credentials configured:\n'
                       'Slack enabled, id: "SMOKE-slack-0001"',
        "user_id": "U000SMOKE",
        "minutes_saved": "30",
    }]},
    timeout=600,
)
print(_resp.status_code)
print(json.dumps(_resp.json(), indent=1)[:1500])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cut over
# MAGIC 1. **Give the Workflow Builder node a credential.** It is a native
# MAGIC    Databricks node, so it needs a `databricksApi` (or
# MAGIC    `databricksOAuth2Api`) credential — a DIFFERENT type from the
# MAGIC    `databricks` credential the chat-model nodes use, so the existing
# MAGIC    "SP Prod - Priority Operations" one will not appear in its picker. At
# MAGIC    time of writing the only `databricksApi` credential on the instance is
# MAGIC    "Databricks HR-Dev", which lives in a personal project and points at a
# MAGIC    different workspace; create one in Domain/priority-ops for the
# MAGIC    workspace this notebook deploys to.
# MAGIC 2. Confirm the node's endpointName matches ENDPOINT_NAME above.
# MAGIC 3. Run it end to end from Slack and check both paths: a buildable request
# MAGIC    (status=workflow) and a vague one (status=question, relayed to the user
# MAGIC    by Send a message12).
# MAGIC 4. Only then unpublish the old `automation-builder`.
# MAGIC
# MAGIC Worth watching on the first real runs: `repair_rounds` in the response.
# MAGIC Consistently hitting the cap (3) means the repair loop is churning rather
# MAGIC than converging, which is exactly the failure the benchmark's execution
# MAGIC checker showed — visible here as a number rather than as mystery latency.
