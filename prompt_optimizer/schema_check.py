"""
Pure-Python port of n8n_schema_check/check_params.js.

Same four checks, same output structure, no Node.js subprocess:
  1. unknownParams          — a parameter KEY not declared at its actual
                              location in the node type's schema.
  2. invalidValues          — a declared key whose VALUE isn't one of that
                              field's real allowed values.
  3. danglingNodeReferences — $('Name')/$node['Name']/$items('Name') pointing
                              at a node that doesn't exist in the workflow.
  4. unknownNodeTypes /     — an invented "n8n-nodes-base.*" type, or a
     unknownTypeVersions      typeVersion the installed package doesn't know.

WHY A PORT: the pipeline is being packaged as a Databricks Model Serving
endpoint, and a Node subprocess inside a serving container means shipping a
Node runtime and node_modules in the image and paying process-spawn cost per
request. Everything here is dictionary walking — there was never a reason for
it to be a second language, beyond the schema source happening to be an npm
package.

FAITHFULNESS: function names, structure and comments deliberately mirror the
JS one-for-one so the two can be diffed by eye, and equivalence is asserted
mechanically (see the equivalence harness) rather than assumed. The JS file
stays in the repo as the reference implementation until that harness passes
on a real corpus.

ONE SUBTLETY WORTH NAMING: JavaScript's Set preserves insertion order and
Python's does not, and `validValues` in the output is a materialised list of
one. Using a plain Python set would produce equivalent findings in a
different ORDER, which would fail byte-comparison and, worse, make the
error messages non-deterministic run to run. So every ordered value-set here
is a dict-with-None-values (insertion-ordered, O(1) membership) and
`list(...)` of it reproduces JS's ordering exactly.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

_HERE = Path(__file__).parent
_SCHEMA_DIR = _HERE / "n8n_schema_check"

# Search order for the node manifest: an installed node_modules first (dev
# machines / the benchmark), then a bundled copy (the serving image, which has
# no npm install). Bundling is what makes this deployable at all.
_NODES_JSON_CANDIDATES = [
    _SCHEMA_DIR / "node_modules" / "n8n-nodes-base" / "dist" / "types" / "nodes.json",
    _SCHEMA_DIR / "nodes.json",
]
_POLLING_JSON = _SCHEMA_DIR / "common_polling_parameters.json"
_PACKAGE_JSON_CANDIDATES = [
    _SCHEMA_DIR / "node_modules" / "n8n-nodes-base" / "package.json",
]


def _ordered_set(values) -> Dict[Any, None]:
    """JS `new Set([...])` — unique, insertion-ordered. See module docstring."""
    return dict.fromkeys(values)


# --------------------------------------------------------------------------
# Manifest loading
# --------------------------------------------------------------------------

_known_base_nodes: Optional[Dict[str, List[Any]]] = None
_node_descriptions_by_name: Optional[Dict[str, List[dict]]] = None
_common_polling_parameters: List[dict] = []
_installed_base_version = "unknown"
_manifest_load_error: Optional[Exception] = None
_loaded = False


def _load_manifest() -> None:
    global _known_base_nodes, _node_descriptions_by_name, _common_polling_parameters
    global _installed_base_version, _manifest_load_error, _loaded
    if _loaded:
        return
    _loaded = True
    path = next((p for p in _NODES_JSON_CANDIDATES if p.exists()), None)
    if path is None:
        _manifest_load_error = FileNotFoundError(
            f"nodes.json not found in any of: "
            f"{', '.join(str(p) for p in _NODES_JSON_CANDIDATES)}"
        )
        return
    try:
        entries = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        _manifest_load_error = e
        return

    # Versioned nodes appear as MULTIPLE entries sharing one name (SlackV1 and
    # SlackV2 are separate entries both named "slack") — union them, or half a
    # node's real versions would be reported as invented.
    known: Dict[str, List[Any]] = {}
    descs: Dict[str, List[dict]] = {}
    for n in entries:
        if not isinstance(n, dict):
            continue
        name = n.get("name")
        version = n.get("version", 1)
        known.setdefault(name, []).extend(version if isinstance(version, list) else [version])
        descs.setdefault(name, []).append(n)
    _known_base_nodes, _node_descriptions_by_name = known, descs

    try:
        _common_polling_parameters = json.loads(_POLLING_JSON.read_text())
    except (OSError, ValueError):
        # Missing/renamed — degrade to not knowing about pollTimes rather than
        # crashing the whole check over one shared constant (as the JS does).
        _common_polling_parameters = []

    for p in _PACKAGE_JSON_CANDIDATES:
        try:
            _installed_base_version = json.loads(p.read_text()).get("version", "unknown")
            break
        except (OSError, ValueError):
            continue


def _max_version(v) -> float:
    if isinstance(v, list):
        return max(float(x) for x in v)
    return float(v)


def find_node_description(short_name: str, type_version) -> Optional[dict]:
    """Entry whose declared version(s) include type_version. On no exact match,
    falls back to the entry with the HIGHEST known version — NOT last-in-array,
    which is arbitrary ordering and once made Slack 2.5 fall back to v1's schema
    (missing "select"), producing a false "unknown parameter" on a real field."""
    entries = (_node_descriptions_by_name or {}).get(short_name)
    if not entries:
        return None
    for e in entries:
        v = e.get("version")
        try:
            if isinstance(v, list):
                if any(float(x) == float(type_version) for x in v):
                    return e
            elif float(v) == float(type_version):
                return e
        except (TypeError, ValueError):
            continue
    best = entries[0]
    for e in entries[1:]:
        try:
            if _max_version(e.get("version")) > _max_version(best.get("version")):
                best = e
        except (TypeError, ValueError):
            continue
    return best


# --------------------------------------------------------------------------
# Composite type schemas (see the JS file's descriptor documentation)
# --------------------------------------------------------------------------

FILTER_COMBINATORS = _ordered_set(["and", "or"])
FILTER_OPERATOR_TYPES = _ordered_set(
    ["string", "number", "boolean", "array", "object", "dateTime", "any"])

_UNIVERSAL = ["exists", "notExists"]
FILTER_OPERATOR_OPERATIONS = {
    "string": _ordered_set(_UNIVERSAL + ["empty", "notEmpty", "equals", "notEquals",
        "contains", "notContains", "startsWith", "notStartsWith", "endsWith",
        "notEndsWith", "regex", "notRegex"]),
    "number": _ordered_set(_UNIVERSAL + ["empty", "notEmpty", "equals", "notEquals",
        "gt", "lt", "gte", "lte"]),
    "dateTime": _ordered_set(_UNIVERSAL + ["empty", "notEmpty", "equals", "notEquals",
        "after", "before", "afterOrEquals", "beforeOrEquals"]),
    "boolean": _ordered_set(_UNIVERSAL + ["empty", "notEmpty", "true", "false",
        "equals", "notEquals"]),
    "array": _ordered_set(_UNIVERSAL + ["contains", "notContains", "lengthEquals",
        "lengthNotEquals", "lengthGt", "lengthLt", "lengthGte", "lengthLte",
        "empty", "notEmpty"]),
    "object": _ordered_set(_UNIVERSAL + ["empty", "notEmpty"]),
    # "any" intentionally absent — operation is left unvalidated for it.
}


def validate_filter_condition(condition, path, value_issues) -> None:
    if not isinstance(condition, dict):
        return
    operator = condition.get("operator")
    if not isinstance(operator, dict):
        return
    type_ = operator.get("type")
    if isinstance(type_, str) and not type_.startswith("=") and type_ not in FILTER_OPERATOR_TYPES:
        value_issues.append({"path": f"{path}.operator.type", "value": type_,
                             "validValues": list(FILTER_OPERATOR_TYPES)})
    operation = operator.get("operation")
    if (isinstance(operation, str) and not operation.startswith("=")
            and isinstance(type_, str) and type_ != "any"):
        valid_ops = FILTER_OPERATOR_OPERATIONS.get(type_)
        if valid_ops and operation not in valid_ops:
            value_issues.append({"path": f"{path}.operator.operation", "value": operation,
                                 "validValues": list(valid_ops)})


FILTER_SCHEMA = {
    "kind": "object",
    "children": {
        "options": {"kind": "object", "children": {
            "caseSensitive": {"kind": "leaf"},
            "leftValue": {"kind": "leaf"},
            "typeValidation": {"kind": "enum", "values": _ordered_set(["strict", "loose"])},
            "version": {"kind": "leaf"},
        }},
        "conditions": {"kind": "array", "item": {
            "kind": "object",
            "children": {
                "id": {"kind": "leaf"},
                "leftValue": {"kind": "leaf"},
                "rightValue": {"kind": "leaf"},
                "operator": {"kind": "object", "children": {
                    "type": {"kind": "enum", "values": FILTER_OPERATOR_TYPES},
                    # operation's valid set depends on the sibling "type" —
                    # handled by customValidate on the parent condition.
                    "operation": {"kind": "leaf"},
                    "rightType": {"kind": "enum", "values": FILTER_OPERATOR_TYPES},
                    "singleValue": {"kind": "leaf"},
                }},
            },
            "customValidate": validate_filter_condition,
        }},
        "combinator": {"kind": "enum", "values": FILTER_COMBINATORS},
    },
}

RESOURCE_MAPPER_SCHEMA = {
    "kind": "object",
    "children": {
        "mappingMode": {"kind": "leaf"},
        "value": {"kind": "opaque"},
        "matchingColumns": {"kind": "leaf"},
        "schema": {"kind": "array", "item": {"kind": "object", "children": {
            "id": {"kind": "leaf"}, "displayName": {"kind": "leaf"},
            "defaultMatch": {"kind": "leaf"}, "canBeUsedToMatch": {"kind": "leaf"},
            "required": {"kind": "leaf"}, "display": {"kind": "leaf"},
            "type": {"kind": "leaf"}, "removed": {"kind": "leaf"},
            "options": {"kind": "opaque"}, "readOnly": {"kind": "leaf"},
            "defaultValue": {"kind": "leaf"},
        }}},
        "attemptToConvertTypes": {"kind": "leaf"},
        "convertFieldsToString": {"kind": "leaf"},
    },
}

ASSIGNMENT_COLLECTION_SCHEMA = {
    "kind": "object",
    "children": {
        "assignments": {"kind": "array", "item": {"kind": "object", "children": {
            "id": {"kind": "leaf"}, "name": {"kind": "leaf"},
            "value": {"kind": "opaque"}, "type": {"kind": "leaf"},
        }}},
    },
}

COMPOSITE_TYPE_SCHEMAS = {
    "filter": FILTER_SCHEMA,
    "resourceMapper": RESOURCE_MAPPER_SCHEMA,
    "assignmentCollection": ASSIGNMENT_COLLECTION_SCHEMA,
}


def describe_resource_locator(prop: dict) -> dict:
    modes = prop.get("modes")
    mode_names = [m.get("name") for m in modes] if isinstance(modes, list) else None
    return {
        "kind": "object",
        "children": {
            "__rl": {"kind": "leaf"},
            "value": {"kind": "opaque"},
            "mode": ({"kind": "enum", "values": _ordered_set(mode_names)}
                     if mode_names else {"kind": "leaf"}),
            "cachedResultName": {"kind": "leaf"},
            "cachedResultUrl": {"kind": "leaf"},
        },
    }


def has_dynamic_options(prop: dict) -> bool:
    to = prop.get("typeOptions")
    if not isinstance(to, dict):
        return False
    return bool(to.get("loadOptionsMethod") or to.get("loadOptionsDependsOn")
                or to.get("searchListMethod"))


def describe_property(prop: dict) -> dict:
    ptype = prop.get("type")
    options = prop.get("options")
    if ptype == "collection" and isinstance(options, list):
        return {"kind": "object", "children": build_child_map(options)}
    if ptype == "fixedCollection" and isinstance(options, list):
        to = prop.get("typeOptions")
        multiple = bool(isinstance(to, dict) and to.get("multipleValues"))
        children = {}
        for group in options:
            group_schema = {"kind": "object",
                            "children": build_child_map(group.get("values") or [])}
            children[group.get("name")] = ({"kind": "array", "item": group_schema}
                                           if multiple else group_schema)
        return {"kind": "object", "children": children}
    if ptype == "resourceLocator":
        return describe_resource_locator(prop)
    if ptype in COMPOSITE_TYPE_SCHEMAS:
        return COMPOSITE_TYPE_SCHEMAS[ptype]
    if ptype in ("options", "multiOptions") and isinstance(options, list) and not has_dynamic_options(prop):
        values = _ordered_set(o["value"] for o in options if isinstance(o, dict) and "value" in o)
        enum_kind = "enumArray" if ptype == "multiOptions" else "enum"
        # Resource/operation-style fields are declared MULTIPLE times under the
        # same name, each gated by displayOptions.show on a sibling. A plain
        # union would accept a value valid only for a DIFFERENT resource, so
        # each declaration's gate is preserved as a branch and resolved against
        # the value's actual siblings at validation time.
        show = (prop.get("displayOptions") or {}).get("show")
        if isinstance(show, dict):
            when = {}
            for field, allowed in show.items():
                if field == prop.get("name") or not isinstance(allowed, list):
                    continue
                when[field] = allowed
            if when:
                return {"kind": "conditionalEnum", "enumKind": enum_kind,
                        "branches": [{"when": when, "values": values}]}
        return {"kind": enum_kind, "values": values}
    return {"kind": "leaf"}


def is_primitive_kind(d: dict) -> bool:
    return d["kind"] in ("leaf", "enum")


def is_enumish(d: dict) -> bool:
    return d["kind"] in ("enum", "enumArray", "conditionalEnum")


def branches_of(d: dict) -> List[dict]:
    if d["kind"] == "conditionalEnum":
        return d["branches"]
    return [{"when": {}, "values": d["values"]}]


def merge_descriptors(a: Optional[dict], b: Optional[dict]) -> Optional[dict]:
    if not a:
        return b
    if not b:
        return a
    if a["kind"] == "opaque" or b["kind"] == "opaque":
        return {"kind": "opaque"}
    if a["kind"] == "conditionalEnum" or b["kind"] == "conditionalEnum":
        if not is_enumish(a) or not is_enumish(b):
            if a["kind"] == "leaf" or b["kind"] == "leaf":
                return {"kind": "leaf"}
            return {"kind": "opaque"}
        enum_kind = ("enumArray"
                     if (a.get("enumKind") == "enumArray" or b.get("enumKind") == "enumArray"
                         or a["kind"] == "enumArray" or b["kind"] == "enumArray")
                     else "enum")
        return {"kind": "conditionalEnum", "enumKind": enum_kind,
                "branches": branches_of(a) + branches_of(b)}
    if a["kind"] != b["kind"]:
        obj = a if a["kind"] == "object" else (b if b["kind"] == "object" else None)
        if obj is not None:
            other = b if obj is a else a
            if other["kind"] == "variant":
                return {"kind": "variant",
                        "objectDesc": merge_descriptors(obj, other["objectDesc"]),
                        "primitiveDesc": other["primitiveDesc"]}
            if is_primitive_kind(other):
                return {"kind": "variant", "objectDesc": obj, "primitiveDesc": other}
        if a["kind"] == "variant" or b["kind"] == "variant":
            variant = a if a["kind"] == "variant" else b
            other_side = b if variant is a else a
            if is_primitive_kind(other_side):
                return {"kind": "variant", "objectDesc": variant["objectDesc"],
                        "primitiveDesc": merge_descriptors(variant["primitiveDesc"], other_side)}
            return {"kind": "opaque"}
        if is_primitive_kind(a) and is_primitive_kind(b):
            return {"kind": "leaf"}
        return {"kind": "opaque"}
    if a["kind"] == "variant":
        return {"kind": "variant",
                "objectDesc": merge_descriptors(a["objectDesc"], b["objectDesc"]),
                "primitiveDesc": merge_descriptors(a["primitiveDesc"], b["primitiveDesc"])}
    if a["kind"] == "object":
        children = dict(a["children"])
        for k, v in b["children"].items():
            children[k] = merge_descriptors(children.get(k), v)
        merged = {"kind": "object", "children": children}
        cv = a.get("customValidate") or b.get("customValidate")
        if cv:
            merged["customValidate"] = cv
        return merged
    if a["kind"] == "array":
        return {"kind": "array", "item": merge_descriptors(a["item"], b["item"])}
    if a["kind"] in ("enum", "enumArray"):
        return {"kind": a["kind"], "values": _ordered_set(list(a["values"]) + list(b["values"]))}
    return a


def build_child_map(props) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for prop in props or []:
        if not isinstance(prop, dict):
            continue
        name = prop.get("name")
        out[name] = merge_descriptors(out.get(name), describe_property(prop))
    return out


def find_mismatches(schema, value, path, issues, value_issues) -> None:
    if not schema or schema["kind"] == "opaque":
        return
    if value is None:
        return
    if schema["kind"] == "leaf":
        return
    if schema["kind"] == "variant":
        if isinstance(value, dict):
            find_mismatches(schema["objectDesc"], value, path, issues, value_issues)
        elif not isinstance(value, list):
            find_mismatches(schema["primitiveDesc"], value, path, issues, value_issues)
        return
    if schema["kind"] == "enum":
        # NOTE: bool is a subclass of int in Python but never a str, so the
        # isinstance check below matches JS's `typeof value !== "string"`.
        if not isinstance(value, str) or value.startswith("="):
            return
        if value not in schema["values"]:
            value_issues.append({"path": path, "value": value,
                                 "validValues": list(schema["values"])})
        return
    if schema["kind"] == "enumArray":
        if not isinstance(value, list):
            return
        for v in value:
            if not isinstance(v, str) or v.startswith("="):
                continue
            if v not in schema["values"]:
                value_issues.append({"path": path, "value": v,
                                     "validValues": list(schema["values"])})
        return
    if schema["kind"] == "array":
        if not isinstance(value, list):
            return
        for item in value:
            find_mismatches(schema["item"], item, path, issues, value_issues)
        return
    if schema["kind"] == "object":
        if not isinstance(value, dict):
            return
        cv = schema.get("customValidate")
        if cv:
            cv(value, path, value_issues)
        for key, child_value in value.items():
            child_schema = schema["children"].get(key)
            if not child_schema:
                issues.append(f"{path}.{key}" if path else key)
                continue
            if child_schema["kind"] == "conditionalEnum":
                validate_conditional_enum(child_schema, child_value, value,
                                          f"{path}.{key}" if path else key, value_issues)
                continue
            find_mismatches(child_schema, child_value,
                            f"{path}.{key}" if path else key, issues, value_issues)


def validate_conditional_enum(schema, value, parent_value, path, value_issues) -> None:
    """Checks only against branches whose discriminator matches the SIBLING
    field(s). Fails open when a discriminator is missing/dynamic, or when no
    branch is confidently active — matching the JS bias toward missed findings
    over false positives."""
    is_array_kind = schema.get("enumKind") == "enumArray"
    if is_array_kind:
        literal_values = ([v for v in value if isinstance(v, str) and not v.startswith("=")]
                          if isinstance(value, list) else [])
    else:
        literal_values = ([value] if isinstance(value, str) and not value.startswith("=") else [])
    if not literal_values:
        return

    matching = []
    for branch in schema["branches"]:
        ok = True
        for field, allowed in branch["when"].items():
            actual = parent_value.get(field) if isinstance(parent_value, dict) else None
            if isinstance(actual, str) and actual not in allowed:
                ok = False
                break
        if ok:
            matching.append(branch)
    if not matching:
        return

    allowed_values: Dict[Any, None] = {}
    for branch in matching:
        for v in branch["values"]:
            allowed_values[v] = None
    for v in literal_values:
        if v not in allowed_values:
            value_issues.append({"path": path, "value": v,
                                 "validValues": list(allowed_values)})


# --------------------------------------------------------------------------
# Dangling node-name references
# --------------------------------------------------------------------------

NODE_REF_PATTERNS = [
    re.compile(r"""\$\(\s*(['"])((?:(?!\1)[\s\S])*)\1\s*\)"""),
    re.compile(r"""\$node\[\s*(['"])((?:(?!\1)[\s\S])*)\1\s*\]"""),
    re.compile(r"""\$items\(\s*(['"])((?:(?!\1)[\s\S])*)\1"""),
]


def collect_node_name_references(value, key, refs: Dict[str, None]) -> None:
    if isinstance(value, str):
        if value.startswith("=") or key == "jsCode":
            for pattern in NODE_REF_PATTERNS:
                for match in pattern.finditer(value):
                    refs[match.group(2)] = None
        return
    if isinstance(value, list):
        for item in value:
            collect_node_name_references(item, key, refs)
        return
    if isinstance(value, dict):
        for k, v in value.items():
            collect_node_name_references(v, k, refs)


def check_node(node: dict, all_node_names: Set[str]) -> Optional[dict]:
    finding = {"node": node.get("name"), "type": node.get("type"),
               "unknownParams": [], "invalidValues": [], "danglingNodeReferences": []}

    node_type = node.get("type")
    short_name = (node_type[len("n8n-nodes-base."):]
                  if isinstance(node_type, str) and node_type.startswith("n8n-nodes-base.")
                  else None)
    desc = (find_node_description(short_name, node.get("typeVersion"))
            if short_name and _node_descriptions_by_name else None)
    if desc:
        top_level_children = build_child_map(desc.get("properties"))
        if desc.get("polling"):
            for k, v in build_child_map(_common_polling_parameters).items():
                top_level_children[k] = merge_descriptors(top_level_children.get(k), v)
        top_level_schema = {"kind": "object", "children": top_level_children}
        issues: List[str] = []
        value_issues: List[dict] = []
        find_mismatches(top_level_schema, node.get("parameters"), "", issues, value_issues)
        # Report just the leaf key name (matching the JS output format).
        finding["unknownParams"] = list(dict.fromkeys(p.split(".")[-1] for p in issues))
        finding["invalidValues"] = value_issues

    refs: Dict[str, None] = {}
    collect_node_name_references(node.get("parameters"), None, refs)
    finding["danglingNodeReferences"] = [n for n in refs if n not in all_node_names]

    if (finding["unknownParams"] or finding["invalidValues"]
            or finding["danglingNodeReferences"]):
        return finding
    return None


def check_node_type_exists(node: dict) -> Optional[dict]:
    if not _known_base_nodes or not isinstance(node.get("type"), str):
        return None
    node_type = node["type"]
    if not node_type.startswith("n8n-nodes-base."):
        return None
    short_name = node_type[len("n8n-nodes-base."):]
    if short_name not in _known_base_nodes:
        return {"kind": "unknownType", "node": node.get("name"), "type": node_type}
    type_version = node.get("typeVersion")
    if type_version is not None:
        versions = _known_base_nodes[short_name]
        try:
            matched = any(float(v) == float(type_version) for v in versions)
        except (TypeError, ValueError):
            matched = False
        if not matched:
            return {"kind": "unknownVersion", "node": node.get("name"), "type": node_type,
                    "typeVersion": type_version, "knownVersions": versions,
                    "installedPackage": f"n8n-nodes-base@{_installed_base_version}"}
    return None


def check_workflow(workflow: dict) -> dict:
    """Same output shape as the JS script's stdout JSON."""
    _load_manifest()
    if _manifest_load_error is not None:
        return {"issues": [], "warnings": [],
                "setupError": f"node schema manifest unavailable: {_manifest_load_error}"}

    issues, unknown_node_types, unknown_type_versions, warnings = [], [], [], []
    nodes = workflow.get("nodes") or []
    all_node_names = {n.get("name") for n in nodes if isinstance(n, dict) and n.get("name")}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        try:
            type_finding = check_node_type_exists(node)
            if type_finding and type_finding["kind"] == "unknownType":
                unknown_node_types.append(type_finding)
            elif type_finding:
                unknown_type_versions.append(type_finding)
            finding = check_node(node, all_node_names)
            if finding:
                issues.append(finding)
        except Exception as e:  # per-node, keep checking the rest
            warnings.append(f"{node.get('name')} ({node.get('type')}): {e}")

    return {
        "issues": [{"node": f["node"], "type": f["type"], "unknownParams": f["unknownParams"]}
                   for f in issues if f["unknownParams"]],
        "invalidValues": [{"node": f["node"], "type": f["type"], "invalidValues": f["invalidValues"]}
                          for f in issues if f["invalidValues"]],
        "danglingNodeReferences": [{"node": f["node"], "type": f["type"],
                                    "danglingNodeReferences": f["danglingNodeReferences"]}
                                   for f in issues if f["danglingNodeReferences"]],
        "unknownNodeTypes": unknown_node_types,
        "unknownTypeVersions": unknown_type_versions,
        "warnings": warnings,
    }
