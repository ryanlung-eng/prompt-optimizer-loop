"""
Deterministic structural validation of the KA's final output — no LLM call.

Answers the narrow question "is this valid, executable n8n JSON?" directly,
as a hard pass/fail per check, rather than an LLM's subjective read of the
response text. Mirrors the checks built earlier for the real n8n
Validator Parser / Code in JavaScript nodes in the automation-builder workflow.
"""
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Neither n8n's own NodeHelpers.getNodeParametersIssues nor workflow
# activation catch invented/hallucinated parameter names (verified directly
# against n8n-workflow/n8n-nodes-base source) — both only flag missing
# *required* fields, since an unrecognized key is just silently ignored at
# runtime rather than erroring. check_params.js compares against each node
# type's own declared schema instead, which does catch it. Optional: skipped
# (not penalized) if Node.js/the npm deps aren't set up.
#
# Setup: copy check_params.js + package.json to local scratch space (e.g.
# /tmp/n8n_schema_check_cache/, the same path _LOCAL_CACHE_DIR below expects)
# and run `npm install --ignore-scripts` there. Deliberately NOT inside this
# module's own directory if that lives in a git-synced Databricks Workspace
# folder — node_modules' ~67k files/symlinks (npm's node_modules/.bin/*
# pattern) can break Databricks Repos' git-status UI, and a network-backed
# /Workspace filesystem makes every cold require() dramatically slower than
# local disk (this is what caused the check to time out before this comment
# was written). This file itself (check_params.js) still lives in the repo,
# tracked and pulled normally — only its node_modules needs to live outside it.
_SCHEMA_CHECK_SCRIPT = Path(__file__).parent / "n8n_schema_check" / "check_params.js"
_SCHEMA_CHECK_TIMEOUT = 30
_node_unavailable: Optional[bool] = None
_timeout_warned = False

# Preferred source for _local_script_path() below — if setup followed the
# instructions above, this already has node_modules installed and the copy
# step is skipped entirely. Falls back to copying from _SCHEMA_CHECK_SCRIPT's
# own directory (the Workspace-synced repo) only if nothing is here yet —
# that fallback existing is what caused the git-status/UI breakage, so it's
# a safety net for un-set-up environments, not the intended steady state.
_LOCAL_CACHE_DIR = Path(tempfile.gettempdir()) / "n8n_schema_check_cache"


def _local_script_path() -> Path:
    local_script = _LOCAL_CACHE_DIR / "check_params.js"
    if not local_script.exists():
        try:
            shutil.copytree(_SCHEMA_CHECK_SCRIPT.parent, _LOCAL_CACHE_DIR, dirs_exist_ok=True)
        except OSError as e:
            print(f"  Warning: couldn't copy n8n schema check to local disk ({e}) — "
                  f"running from its original (possibly slower) location instead.")
            return _SCHEMA_CHECK_SCRIPT
    return local_script


_EMPTY_SCHEMA_FINDINGS = {
    "unknown_params": [], "invalid_values": [], "dangling_refs": [],
    "unknown_node_types": [], "unknown_type_versions": [],
}


def _check_schema_issues(workflow: dict) -> dict:
    """
    Runs check_params.js and returns its three finding categories as
    {"unknown_params": [...], "invalid_values": [...], "dangling_refs": [...]},
    each a list of human-readable error strings. All empty when the check is
    unavailable (Node.js/npm deps missing) — unavailable is "skipped", never
    "failed".
    """
    global _node_unavailable, _timeout_warned
    if _node_unavailable:
        return _EMPTY_SCHEMA_FINDINGS
    try:
        proc = subprocess.run(
            ["node", str(_local_script_path())],
            input=json.dumps(workflow),
            capture_output=True,
            text=True,
            timeout=_SCHEMA_CHECK_TIMEOUT,
        )
    except FileNotFoundError as e:
        # The node binary itself missing won't change mid-run — stop trying
        # entirely instead of re-attempting (and re-printing) on every call.
        if _node_unavailable is None:
            print(f"  Warning: n8n schema parameter check unavailable ({e}) — skipping. "
                  f"Setup: install Node.js, then see the setup comment above "
                  f"_LOCAL_CACHE_DIR in this file (prompt_optimizer/validator.py).")
        _node_unavailable = True
        return _EMPTY_SCHEMA_FINDINGS
    except subprocess.TimeoutExpired:
        # A slow filesystem or a one-off hiccup isn't the same as "broken" —
        # skip just this call and keep retrying later ones, rather than
        # disabling the check for the rest of the run over one slow call.
        if not _timeout_warned:
            print(f"  Warning: n8n schema parameter check timed out "
                  f"(>{_SCHEMA_CHECK_TIMEOUT}s) for one workflow — skipping just "
                  f"this check; will keep retrying on later calls.")
            _timeout_warned = True
        return _EMPTY_SCHEMA_FINDINGS

    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        _node_unavailable = False
        return _EMPTY_SCHEMA_FINDINGS

    setup_error = result.get("setupError")
    if setup_error:
        # Same failure every call (missing npm deps, not a per-node issue) —
        # print once and stop spawning the subprocess for the rest of the
        # run, instead of repeating this per node per turn per input.
        if _node_unavailable is None:
            print(f"  Warning: n8n schema parameter check unavailable — {setup_error}")
        _node_unavailable = True
        return _EMPTY_SCHEMA_FINDINGS

    _node_unavailable = False
    for w in result.get("warnings") or []:
        print(f"  Warning: n8n schema check couldn't fully validate node {w}")

    unknown_params = [
        f"Node '{issue['node']}' ({issue['type']}) uses parameter(s) not in n8n's "
        f"schema for that node type — likely invented/hallucinated: {issue['unknownParams']}"
        for issue in result.get("issues") or []
    ]
    invalid_values = [
        f"Node '{issue['node']}' ({issue['type']}) uses value(s) not in the real "
        f"allowed set for their field — likely invented/hallucinated: "
        + "; ".join(
            f"{v['path']}={v['value']!r} (valid: {v['validValues']})"
            for v in issue["invalidValues"]
        )
        for issue in result.get("invalidValues") or []
    ]
    dangling_refs = [
        f"Node '{issue['node']}' ({issue['type']}) has expression(s) referencing "
        f"node name(s) that don't exist in this workflow: {issue['danglingNodeReferences']} "
        f"— these evaluate to nothing at runtime."
        for issue in result.get("danglingNodeReferences") or []
    ]
    unknown_node_types = [
        f"Node '{issue['node']}' claims node type '{issue['type']}', which does not "
        f"exist in n8n-nodes-base at all — an invented node type."
        for issue in result.get("unknownNodeTypes") or []
    ]
    unknown_type_versions = [
        f"Node '{issue['node']}' ({issue['type']}) uses typeVersion "
        f"{issue['typeVersion']}, which is not a real version of that node "
        f"(known: {issue['knownVersions']}, per {issue.get('installedPackage', 'installed package')})."
        for issue in result.get("unknownTypeVersions") or []
    ]
    return {
        "unknown_params": unknown_params,
        "invalid_values": invalid_values,
        "dangling_refs": dangling_refs,
        "unknown_node_types": unknown_node_types,
        "unknown_type_versions": unknown_type_versions,
    }


# ---------------------------------------------------------------------------
# Graph-level checks — pure JSON analysis, no schema/Node.js needed. These
# catch workflows whose every individual node is syntactically perfect but
# whose WIRING makes them broken or inert at runtime.
# ---------------------------------------------------------------------------

# Keys whose string values are raw code, not n8n expression strings — a
# "{{ $json.x }}" inside JavaScript source is not an n8n expression and must
# not be held to the "=" prefix rule.
_CODE_KEYS = {"jsCode", "pythonCode", "functionCode"}

# A string containing "{{ ... $... }}" is an attempted n8n expression — n8n
# only EVALUATES it if the string starts with "=" (otherwise it's sent as
# the literal text "{{ $json.x }}"), a silent, high-frequency failure mode
# called out in the platform's own docs. Requiring a "$" inside the braces
# keeps this fair: literal prose braces without n8n variables don't flag.
_EXPR_PATTERN = re.compile(r"\{\{[^}]*\$")


def _find_unprefixed_expressions(params, key=None, path="", found=None) -> List[str]:
    if found is None:
        found = []
    if isinstance(params, str):
        if key not in _CODE_KEYS and _EXPR_PATTERN.search(params) and not params.startswith("="):
            found.append(f"{path}: {params[:80]!r}")
    elif isinstance(params, list):
        for i, item in enumerate(params):
            _find_unprefixed_expressions(item, key, f"{path}[{i}]", found)
    elif isinstance(params, dict):
        for k, v in params.items():
            _find_unprefixed_expressions(v, k, f"{path}.{k}" if path else k, found)
    return found


# n8n's Schedule Trigger accepts standard 5-field cron plus an optional
# seconds field (6 fields). Field content is validated loosely (numbers,
# ranges, steps, month/day names, wildcards) — the goal is catching prose or
# structurally impossible strings, not reimplementing a cron parser.
_CRON_FIELD = re.compile(r"^[A-Za-z0-9*,/?LW#-]+$")


def _check_cron_expressions(nodes: List[dict]) -> List[str]:
    errors = []
    for n in nodes:
        if not isinstance(n, dict) or n.get("type") not in (
            "n8n-nodes-base.scheduleTrigger", "n8n-nodes-base.cron",
        ):
            continue
        rule = n.get("parameters", {}).get("rule", {})
        intervals = rule.get("interval", []) if isinstance(rule, dict) else []
        for item in intervals if isinstance(intervals, list) else []:
            if not isinstance(item, dict) or item.get("field") != "cronExpression":
                continue
            expr = item.get("expression")
            if not isinstance(expr, str) or expr.startswith("=") or "{{" in expr:
                continue  # expression-valued — not statically checkable
            fields = expr.split()
            if not (5 <= len(fields) <= 6) or not all(_CRON_FIELD.match(f) for f in fields):
                errors.append(
                    f"Node '{n.get('name')}' has an invalid cron expression "
                    f"{expr!r} — expected 5 fields (minute hour day-of-month "
                    f"month day-of-week) or 6 with a leading seconds field."
                )
    return errors


def _iter_connection_targets(connections: dict):
    """Yields (source_name, conn_type, output_index, target_name, input_index)."""
    for src_name, groups in connections.items():
        if not isinstance(groups, dict):
            continue
        for conn_type, output_groups in groups.items():
            if not isinstance(output_groups, list):
                continue
            for out_idx, group in enumerate(output_groups):
                for conn in group if isinstance(group, list) else []:
                    if isinstance(conn, dict) and conn.get("node"):
                        yield src_name, conn_type, out_idx, conn["node"], conn.get("index", 0)


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

    # Source keys must be real node NAMES (a key matching no node usually
    # means the id was used instead of the name — n8n silently ignores the
    # whole entry, disconnecting everything downstream of it).
    bad_sources = [s for s in connections if s not in node_names]
    checks["all_connection_sources_are_real_nodes"] = len(bad_sources) == 0
    if bad_sources:
        result.errors.append(
            f"Connections keyed by names that match no node (connections must use "
            f"the node 'name' field, never 'id'): {bad_sources}"
        )

    # Targets checked across ALL connection types (main, ai_languageModel,
    # ai_outputParser, ...), not just the main chain.
    dangling_refs = [
        target for _, _, _, target, _ in _iter_connection_targets(connections)
        if target not in node_names
    ]
    checks["all_connections_reference_real_nodes"] = len(dangling_refs) == 0
    if dangling_refs:
        result.errors.append(f"Connections reference nonexistent nodes: {dangling_refs}")

    # Every node must participate in the graph somewhere (as a source or a
    # target of any connection type) — a node in neither is dead weight that
    # never runs. Sticky notes are legitimately unwired; a single-node
    # workflow (bare trigger) has nothing to wire.
    connected = set(connections)
    connected |= {t for _, _, _, t, _ in _iter_connection_targets(connections)}
    disconnected = [
        n.get("name") for n in nodes
        if isinstance(n, dict) and n.get("name") not in connected
        and n.get("type") != "n8n-nodes-base.stickyNote"
    ] if len(nodes) > 1 else []
    checks["no_disconnected_nodes"] = len(disconnected) == 0
    if disconnected:
        result.errors.append(
            f"Nodes not wired into the workflow at all (present in 'nodes' but "
            f"absent from 'connections' as both source and target): {disconnected}"
        )

    # An AI Agent with no model sub-node wired into its ai_languageModel slot
    # fails to publish (platform-verified). The connection direction is
    # sub-node -> agent, so we look for the agent as a TARGET of that type.
    agent_names = {
        n.get("name") for n in nodes
        if isinstance(n, dict) and str(n.get("type", "")).endswith(".agent")
    }
    agents_with_model = {
        target for _, conn_type, _, target, _ in _iter_connection_targets(connections)
        if conn_type == "ai_languageModel"
    }
    agents_missing_model = sorted(agent_names - agents_with_model)
    checks["ai_agents_have_model"] = len(agents_missing_model) == 0
    if agents_missing_model:
        result.errors.append(
            f"AI Agent node(s) with no model sub-node connected via "
            f"ai_languageModel (will fail to publish): {agents_missing_model}"
        )

    # Platform rule (documented as CRITICAL in the KB): a getAll operation
    # feeding an AI Agent directly makes the agent run once per item,
    # duplicating every downstream send. An aggregation Code node must sit
    # between them.
    getall_names = {
        n.get("name") for n in nodes
        if isinstance(n, dict) and n.get("parameters", {}).get("operation") == "getAll"
    }
    unaggregated = sorted({
        f"{src} → {target}"
        for src, conn_type, _, target, _ in _iter_connection_targets(connections)
        if conn_type == "main" and src in getall_names and target in agent_names
    })
    checks["no_unaggregated_getall_into_agent"] = len(unaggregated) == 0
    if unaggregated:
        result.errors.append(
            f"getAll output connected directly into an AI Agent — the agent "
            f"will run once per item, duplicating every downstream action; an "
            f"aggregation Code node must sit between them: {unaggregated}"
        )

    # An IF node routing both its true and false outputs to the same target
    # input does nothing — the branch condition has no effect. (Both outputs
    # into DIFFERENT inputs of e.g. a Merge node is a legitimate pattern and
    # is not flagged.)
    if_names = {
        n.get("name") for n in nodes
        if isinstance(n, dict) and n.get("type") == "n8n-nodes-base.if"
    }
    pointless_ifs = []
    for if_name in if_names:
        groups = connections.get(if_name, {})
        outputs = groups.get("main", []) if isinstance(groups, dict) else []
        if len(outputs) >= 2:
            branch_targets = [
                {
                    (c.get("node"), c.get("index", 0))
                    for c in (out if isinstance(out, list) else []) if isinstance(c, dict)
                }
                for out in outputs[:2]
            ]
            overlap = branch_targets[0] & branch_targets[1]
            if overlap:
                pointless_ifs.append(f"{if_name} (both branches → {sorted(t[0] for t in overlap)})")
    checks["no_pointless_if_branches"] = len(pointless_ifs) == 0
    if pointless_ifs:
        result.errors.append(
            f"IF node(s) route both true AND false branches to the same target "
            f"input — the condition has no effect: {pointless_ifs}"
        )

    # Attempted n8n expressions missing the "=" prefix are sent as literal
    # text instead of being evaluated — a silent failure.
    unprefixed = []
    for n in nodes:
        if isinstance(n, dict):
            hits = _find_unprefixed_expressions(n.get("parameters", {}))
            unprefixed.extend(f"Node '{n.get('name')}' {h}" for h in hits)
    checks["expressions_have_equals_prefix"] = len(unprefixed) == 0
    if unprefixed:
        result.errors.append(
            "Expression string(s) missing the '=' prefix — n8n will send the "
            "literal text '{{ ... }}' instead of evaluating it: " + "; ".join(unprefixed)
        )

    cron_errors = _check_cron_expressions(nodes)
    checks["valid_cron_expressions"] = len(cron_errors) == 0
    result.errors.extend(cron_errors)

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

    schema_findings = _check_schema_issues(workflow)
    checks["no_unknown_parameters"] = len(schema_findings["unknown_params"]) == 0
    checks["no_invalid_parameter_values"] = len(schema_findings["invalid_values"]) == 0
    checks["no_dangling_node_references"] = len(schema_findings["dangling_refs"]) == 0
    checks["no_unknown_node_types"] = len(schema_findings["unknown_node_types"]) == 0
    checks["no_unknown_type_versions"] = len(schema_findings["unknown_type_versions"]) == 0
    for findings in schema_findings.values():
        result.errors.extend(findings)

    return result
