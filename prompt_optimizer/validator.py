"""
Deterministic structural validation of the KA's final output — no LLM call.

Answers the narrow question "is this valid, executable n8n JSON?" directly,
as a hard pass/fail per check, rather than an LLM's subjective read of the
response text. Mirrors the checks built earlier for the real n8n
Validator Parser / Code in JavaScript nodes in the automation-builder workflow.
"""
import json
from dataclasses import dataclass, field
from typing import List


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

    return result
