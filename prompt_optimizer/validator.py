"""
Deterministic structural validation of the KA's final output — no LLM call.

Answers the narrow question "is this valid, executable n8n JSON?" directly,
as a hard pass/fail per check, rather than an LLM's subjective read of the
response text. Mirrors the checks built earlier for the real n8n
Validator Parser / Code in JavaScript nodes in the automation-builder workflow.
"""
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Neither n8n's own NodeHelpers.getNodeParametersIssues nor workflow
# activation catch invented/hallucinated parameter names (verified directly
# against n8n-workflow/n8n-nodes-base source) — both only flag missing
# *required* fields, since an unrecognized key is just silently ignored at
# runtime rather than erroring. check_params.js compares against each node
# type's own declared schema instead, which does catch it. Optional: skipped
# (not penalized) if Node.js/the npm deps aren't set up in this environment —
# see prompt_optimizer/n8n_schema_check/.
_SCHEMA_CHECK_SCRIPT = Path(__file__).parent / "n8n_schema_check" / "check_params.js"
# Generous — Databricks /Workspace paths can be network-backed, making
# n8n-nodes-base's large require() tree noticeably slower to cold-load than
# on local disk (sub-second locally; seen taking >10s there).
_SCHEMA_CHECK_TIMEOUT = 30
_node_unavailable: Optional[bool] = None
_timeout_warned = False


def _check_unknown_parameters(workflow: dict) -> List[str]:
    global _node_unavailable, _timeout_warned
    if _node_unavailable:
        return []
    try:
        proc = subprocess.run(
            ["node", str(_SCHEMA_CHECK_SCRIPT)],
            input=json.dumps(workflow),
            capture_output=True,
            text=True,
            timeout=_SCHEMA_CHECK_TIMEOUT,
        )
    except FileNotFoundError as e:
        # The node binary itself missing won't change mid-run — stop trying
        # entirely instead of re-attempting (and re-printing) on every call.
        if _node_unavailable is None:
            print(f"  Warning: n8n schema parameter check unavailable ({e}) — "
                  f"skipping; see prompt_optimizer/n8n_schema_check/ for setup.")
        _node_unavailable = True
        return []
    except subprocess.TimeoutExpired:
        # A slow filesystem or a one-off hiccup isn't the same as "broken" —
        # skip just this call and keep retrying later ones, rather than
        # disabling the check for the rest of the run over one slow call.
        if not _timeout_warned:
            print(f"  Warning: n8n schema parameter check timed out "
                  f"(>{_SCHEMA_CHECK_TIMEOUT}s) for one workflow — skipping just "
                  f"this check; will keep retrying on later calls.")
            _timeout_warned = True
        return []

    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        _node_unavailable = False
        return []

    setup_error = result.get("setupError")
    if setup_error:
        # Same failure every call (missing npm deps, not a per-node issue) —
        # print once and stop spawning the subprocess for the rest of the
        # run, instead of repeating this per node per turn per input.
        if _node_unavailable is None:
            print(f"  Warning: n8n schema parameter check unavailable — {setup_error}")
        _node_unavailable = True
        return []

    _node_unavailable = False
    for w in result.get("warnings") or []:
        print(f"  Warning: n8n schema check couldn't fully validate node {w}")
    return [
        f"Node '{issue['node']}' ({issue['type']}) uses parameter(s) not in n8n's "
        f"schema for that node type — likely invented/hallucinated: {issue['unknownParams']}"
        for issue in result.get("issues") or []
    ]


@dataclass
class StructuralResult:
    is_json: bool = False
    checks: dict = field(default_factory=dict)   # check_name -> bool
    errors: List[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.is_json and all(self.checks.values())

    @property
    def score(self) -> float:
        """Fraction of checks passed — 0.0 if it isn't even parseable JSON."""
        if not self.is_json or not self.checks:
            return 0.0
        return sum(1 for v in self.checks.values() if v) / len(self.checks)


def validate_workflow_json(text: str) -> StructuralResult:
    result = StructuralResult()
    text = text.strip()

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        result.errors.append("No JSON object found in response — the KA never output a workflow.")
        return result

    try:
        workflow = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        result.errors.append(f"Invalid JSON: {e}")
        return result

    result.is_json = True
    checks = result.checks

    nodes = workflow.get("nodes")
    checks["has_nodes_array"] = isinstance(nodes, list) and len(nodes) > 0
    if not checks["has_nodes_array"]:
        result.errors.append("Missing or empty 'nodes' array.")
        nodes = []

    connections = workflow.get("connections")
    checks["has_connections_object"] = isinstance(connections, dict)
    if not checks["has_connections_object"]:
        result.errors.append("Missing or invalid 'connections' object.")
        connections = {}

    settings = workflow.get("settings", {})
    checks["execution_order_v1"] = settings.get("executionOrder") == "v1"
    if not checks["execution_order_v1"]:
        result.errors.append(f"settings.executionOrder is {settings.get('executionOrder')!r}, expected 'v1'.")

    node_names = {n.get("name") for n in nodes if isinstance(n, dict)}

    # n8n node "id" just needs to be a unique, non-empty string identifier —
    # NOT necessarily UUID-formatted. Confirmed against a real, authoritative
    # example workflow that uses descriptive slug IDs throughout (e.g.
    # "google-sheets-trigger", "get-dm-channel-id") rather than UUIDs.
    has_valid_ids = all(
        isinstance(n.get("id"), str) and n["id"].strip()
        for n in nodes if isinstance(n, dict)
    )
    checks["all_nodes_have_id"] = has_valid_ids if nodes else False
    if nodes and not has_valid_ids:
        result.errors.append("One or more nodes are missing a non-empty 'id'.")

    has_type_version = all(
        "typeVersion" in n for n in nodes if isinstance(n, dict)
    )
    checks["all_nodes_have_type_version"] = has_type_version if nodes else False
    if nodes and not has_type_version:
        result.errors.append("One or more nodes are missing 'typeVersion'.")

    dangling_refs = []
    for src_name, groups in connections.items():
        for group in groups.get("main", []) if isinstance(groups, dict) else []:
            for conn in group if isinstance(group, list) else []:
                target = conn.get("node") if isinstance(conn, dict) else None
                if target and target not in node_names:
                    dangling_refs.append(target)
    checks["all_connections_reference_real_nodes"] = len(dangling_refs) == 0
    if dangling_refs:
        result.errors.append(f"Connections reference nonexistent nodes: {dangling_refs}")

    empty_creds = [
        n.get("name") for n in nodes if isinstance(n, dict)
        for cred in n.get("credentials", {}).values()
        if isinstance(cred, dict) and not cred.get("id")
    ]
    checks["no_empty_credential_ids"] = len(empty_creds) == 0
    if empty_creds:
        result.errors.append(f"Nodes with empty credential IDs: {empty_creds}")

    has_trigger = any(
        isinstance(n, dict) and "trigger" in n.get("type", "").lower()
        for n in nodes
    )
    checks["has_trigger_node"] = has_trigger if nodes else False
    if nodes and not has_trigger:
        result.errors.append("No node type contains 'trigger' — workflow has no entry point.")

    unknown_param_errors = _check_unknown_parameters(workflow)
    checks["no_unknown_parameters"] = len(unknown_param_errors) == 0
    result.errors.extend(unknown_param_errors)

    return result
