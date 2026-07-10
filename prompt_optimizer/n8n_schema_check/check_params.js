#!/usr/bin/env node
/**
 * Reads a workflow JSON on stdin and runs three independent, deterministic
 * checks against each node whose type we have a local n8n-nodes-base
 * definition for, plus one whole-workflow check that applies regardless of
 * node-type recognition:
 *
 *   1. unknownParams — a parameter KEY that doesn't exist at its actual
 *      location in the node type's declared schema (path-aware: a key valid
 *      elsewhere in the same node's schema but not at THIS location is still
 *      flagged — see the path-aware rewrite notes below).
 *   2. invalidValues — a parameter key that DOES exist, but whose VALUE isn't
 *      one of that field's real allowed values (an "options"/"multiOptions"
 *      dropdown's real .options list, or a filter condition's
 *      combinator/operator.type/operator.operation, verified directly
 *      against n8n's actual executor logic — see FILTER_OPERATOR_OPERATIONS
 *      below). Dynamic fields (loadOptionsMethod-backed) and any expression
 *      string (starts with "=") are deliberately never checked here, since
 *      their real valid value either isn't statically knowable (fetched from
 *      an external system at runtime) or isn't known until the expression
 *      itself is evaluated at runtime.
 *   3. danglingNodeReferences — an expression string (or a Code node's
 *      jsCode, which doesn't need the "=" prefix) that references another
 *      node by name (via $('Name'), $node['Name'], or $items('Name')) where
 *      that name doesn't match any real node in the workflow. This check
 *      doesn't need a recognized node type at all, so it applies to every
 *      node, including custom/unrecognized ones.
 *
 * This exists because neither n8n's own NodeHelpers.getNodeParametersIssues
 * nor workflow activation catch any of this (verified empirically) — they
 * only catch missing *required* fields. Since a fabricated parameter/value/
 * reference is silently ignored or silently evaluates to nothing at runtime
 * rather than erroring at generation time, the only way to catch it ahead of
 * time is comparing against the real schema/executor logic directly.
 *
 * PATH-AWARE (unknownParams): walks the node's declared property schema and
 * its actual `parameters` object in parallel, checking each key against the
 * schema's structure at its correct corresponding nesting location — not
 * just "does a key with this name exist somewhere in the whole schema." An
 * earlier flat, name-only version of this check could be fooled by a
 * fabricated key that happened to reuse a valid name from a DIFFERENT
 * location in the same node's schema (e.g. a bogus top-level "leftValue"
 * sibling of "conditions" on an If node — "leftValue" is a real field name,
 * just never valid at the top level, only nested inside
 * conditions.conditions[]).
 *
 * Scope note: this still does NOT model displayOptions-based conditional
 * visibility (e.g. a property only valid for a specific "operation"/
 * "resource" selection). Different operations can declare a same-named
 * property with a different shape (Slack's "channelId" is a plain string
 * for the "create" operation but a resourceLocator object everywhere else);
 * when that happens this check merges the possible shapes permissively
 * rather than picking one and false-flagging the other's valid usage.
 *
 * Only the node types actually used by the Workflow Builder are covered for
 * checks 1-2 — unrecognized types (e.g. Ibotta's own custom KA node) are
 * skipped for those, not flagged as errors, since we have no schema to
 * compare against. Check 3 applies to every node regardless.
 *
 * Output (stdout, always valid JSON):
 *   {"issues": [...], "invalidValues": [...], "danglingNodeReferences": [...], "warnings": [...]}
 */
// n8n's node loader auto-injects a shared "Poll Times" fixedCollection
// (pollTimes -> {item: [{mode, ...}]}) into every node with description
// .polling === true — it's never declared in the individual node's own
// .properties array, which is why GmailTrigger/GoogleCalendarTrigger/etc.
// all show no "pollTimes" property directly even though real generated
// workflows correctly use it. Verified against the actual source:
// n8n-core/dist/nodes-loader/constants.js.
let commonPollingParameters = [];
try {
  ({ commonPollingParameters } = require("n8n-core/dist/nodes-loader/constants.js"));
} catch (e) {
  // Missing/renamed in some n8n-core version — degrade to not knowing about
  // pollTimes rather than crashing the whole check over one shared constant.
}

// Authoritative list of EVERY node type shipped by the installed
// n8n-nodes-base package (485 entries: short name + real version list),
// used to catch entirely-invented node types ("n8n-nodes-base.agent",
// "n8n-nodes-base.databricksCloudModel" — both real hallucinations this
// platform's docs warn about) and invented typeVersions. Only claims about
// the "n8n-nodes-base." prefix are checkable — custom/community prefixes
// (CUSTOM.*, @n8n/n8n-nodes-langchain.*) live outside this package, so
// they're skipped, never flagged.
let knownBaseNodes = null; // Map short-name -> [versions]
try {
  const list = require("n8n-nodes-base/dist/types/nodes.json");
  // Versioned nodes appear as MULTIPLE entries sharing one name (SlackV1 and
  // SlackV2 are separate entries both named "slack", each carrying only its
  // own major's version list) — union them, or half a node's real versions
  // would be reported as invented.
  knownBaseNodes = new Map();
  for (const n of list) {
    const versions = knownBaseNodes.get(n.name) || [];
    knownBaseNodes.set(n.name, versions.concat(n.version ?? 1));
  }
} catch (e) {
  // Older package layout without dist/types/nodes.json — skip these two
  // checks rather than crash the rest.
}
let installedBaseVersion = "unknown";
try {
  installedBaseVersion = require("n8n-nodes-base/package.json").version;
} catch (e) { /* same degradation */ }

const NODE_TYPE_MAP = {
  "n8n-nodes-base.scheduleTrigger": ["n8n-nodes-base/dist/nodes/Schedule/ScheduleTrigger.node.js", "ScheduleTrigger"],
  "n8n-nodes-base.cron": ["n8n-nodes-base/dist/nodes/Cron/Cron.node.js", "Cron"],
  "n8n-nodes-base.gmailTrigger": ["n8n-nodes-base/dist/nodes/Google/Gmail/GmailTrigger.node.js", "GmailTrigger"],
  "n8n-nodes-base.gmail": ["n8n-nodes-base/dist/nodes/Google/Gmail/Gmail.node.js", "Gmail"],
  "n8n-nodes-base.slackTrigger": ["n8n-nodes-base/dist/nodes/Slack/SlackTrigger.node.js", "SlackTrigger"],
  "n8n-nodes-base.slack": ["n8n-nodes-base/dist/nodes/Slack/Slack.node.js", "Slack"],
  "n8n-nodes-base.jiraTrigger": ["n8n-nodes-base/dist/nodes/Jira/JiraTrigger.node.js", "JiraTrigger"],
  "n8n-nodes-base.jira": ["n8n-nodes-base/dist/nodes/Jira/Jira.node.js", "Jira"],
  "n8n-nodes-base.googleSheetsTrigger": ["n8n-nodes-base/dist/nodes/Google/Sheet/GoogleSheetsTrigger.node.js", "GoogleSheetsTrigger"],
  "n8n-nodes-base.googleSheets": ["n8n-nodes-base/dist/nodes/Google/Sheet/GoogleSheets.node.js", "GoogleSheets"],
  "n8n-nodes-base.httpRequest": ["n8n-nodes-base/dist/nodes/HttpRequest/HttpRequest.node.js", "HttpRequest"],
  "n8n-nodes-base.executeWorkflow": ["n8n-nodes-base/dist/nodes/ExecuteWorkflow/ExecuteWorkflow/ExecuteWorkflow.node.js", "ExecuteWorkflow"],
  "n8n-nodes-base.if": ["n8n-nodes-base/dist/nodes/If/If.node.js", "If"],
  "n8n-nodes-base.noOp": ["n8n-nodes-base/dist/nodes/NoOp/NoOp.node.js", "NoOp"],
  "n8n-nodes-base.code": ["n8n-nodes-base/dist/nodes/Code/Code.node.js", "Code"],
  "n8n-nodes-base.set": ["n8n-nodes-base/dist/nodes/Set/Set.node.js", "Set"],
};

// ---------------------------------------------------------------------------
// Schema descriptors.
//
// A descriptor describes what SHAPE a value is allowed to take at one point
// in the parameter tree:
//   { kind: "leaf" }                          - primitive/opaque scalar value
//                                                (string, number, boolean,
//                                                json, dateTime, an
//                                                expression string, ...) - no
//                                                nested keys AND no value
//                                                constraint to check.
//   { kind: "opaque" }                         - deliberately never checked,
//                                                even though it may be an
//                                                object/array (resourceMapper
//                                                "value", a merge conflict -
//                                                see mergeDescriptors).
//   { kind: "enum", values: Set }              - value must be a literal
//                                                (non-expression) string
//                                                that's a member of `values`
//                                                - a real "options"-type
//                                                dropdown's declared choices,
//                                                or a filter operator's
//                                                type/operation/combinator.
//   { kind: "enumArray", values: Set }         - value is an array; every
//                                                literal (non-expression)
//                                                string element must be a
//                                                member of `values` - a
//                                                "multiOptions"-type field.
//   { kind: "object", children: {name: desc},
//     customValidate?: fn }                    - value is an object; every
//                                                key must be a known child,
//                                                each validated against ITS
//                                                OWN descriptor. customValidate,
//                                                when present, is called with
//                                                (value, path, valueIssues)
//                                                for cross-field checks that
//                                                a flat per-key walk can't
//                                                express (e.g. filter's
//                                                operator.operation validity
//                                                depends on the SIBLING
//                                                operator.type field's value).
//   { kind: "array", item: desc }              - value is an array; every
//                                                element validated against
//                                                `item`.
// ---------------------------------------------------------------------------

// n8n's shared composite parameter types (filter, resourceMapper,
// assignmentCollection, resourceLocator) have a FIXED runtime value shape
// defined by n8n's own TypeScript interfaces — FilterValue/
// FilterConditionValue/FilterOperatorValue/FilterOptionsValue,
// ResourceMapperValue/ResourceMapperField, AssignmentCollectionValue/
// AssignmentValue — verified directly against n8n-workflow's
// interfaces.ts, not declared via the property's own .options array the
// way collection/fixedCollection are. These are hand-written, nested
// descriptors (not a flat name list) so a key can only match at its real
// location — e.g. "leftValue"/"operator"/"combinator" are only valid nested
// inside conditions[], never as a top-level sibling of "conditions".

// Combinator and operator.type are CLOSED TypeScript union types in n8n's
// own interfaces.ts (FilterTypeCombinator = 'and' | 'or';
// FilterOperatorType = 'string' | 'number' | 'boolean' | 'array' | 'object'
// | 'dateTime' | 'any') — safe to validate strictly.
const FILTER_COMBINATORS = new Set(["and", "or"]);
const FILTER_OPERATOR_TYPES = new Set(["string", "number", "boolean", "array", "object", "dateTime", "any"]);

// operator.operation is typed as a loose `string` at the interface level
// (n8n allows custom expression strings there too), but the REAL valid,
// non-expression values are fully enumerated by the actual runtime switch
// statement in n8n-workflow's executeFilterCondition (packages/workflow/src/
// node-parameters/filter-parameter.ts) — this is more authoritative than a
// UI-declared list, since a value outside this set literally does nothing
// at execution time regardless of what any dropdown might have offered.
// "exists"/"notExists" are handled BEFORE the type-specific switch in that
// same function, so they're valid for every type. "any" has no case in the
// switch at all (falls through to the "unknown operator" branch) - its
// exact runtime behavior is genuinely undefined/degenerate, so operation is
// deliberately left unchecked for "any" rather than guessing.
const FILTER_OPERATIONS_UNIVERSAL = ["exists", "notExists"];
const FILTER_OPERATOR_OPERATIONS = {
  string: new Set([...FILTER_OPERATIONS_UNIVERSAL, "empty", "notEmpty", "equals", "notEquals",
    "contains", "notContains", "startsWith", "notStartsWith", "endsWith", "notEndsWith",
    "regex", "notRegex"]),
  number: new Set([...FILTER_OPERATIONS_UNIVERSAL, "empty", "notEmpty", "equals", "notEquals",
    "gt", "lt", "gte", "lte"]),
  dateTime: new Set([...FILTER_OPERATIONS_UNIVERSAL, "empty", "notEmpty", "equals", "notEquals",
    "after", "before", "afterOrEquals", "beforeOrEquals"]),
  boolean: new Set([...FILTER_OPERATIONS_UNIVERSAL, "empty", "notEmpty", "true", "false",
    "equals", "notEquals"]),
  array: new Set([...FILTER_OPERATIONS_UNIVERSAL, "contains", "notContains", "lengthEquals",
    "lengthNotEquals", "lengthGt", "lengthLt", "lengthGte", "lengthLte", "empty", "notEmpty"]),
  object: new Set([...FILTER_OPERATIONS_UNIVERSAL, "empty", "notEmpty"]),
  // "any" intentionally has no entry — operation is left unvalidated for it.
};

function validateFilterCondition(condition, path, valueIssues) {
  if (!condition || typeof condition !== "object") return;
  const operator = condition.operator;
  if (!operator || typeof operator !== "object") return;
  const type = operator.type;
  if (typeof type === "string" && !type.startsWith("=") && !FILTER_OPERATOR_TYPES.has(type)) {
    valueIssues.push({ path: `${path}.operator.type`, value: type, validValues: [...FILTER_OPERATOR_TYPES] });
  }
  const operation = operator.operation;
  if (typeof operation === "string" && !operation.startsWith("=") && typeof type === "string" && type !== "any") {
    const validOps = FILTER_OPERATOR_OPERATIONS[type];
    if (validOps && !validOps.has(operation)) {
      valueIssues.push({ path: `${path}.operator.operation`, value: operation, validValues: [...validOps] });
    }
  }
}

const FILTER_SCHEMA = {
  kind: "object",
  children: {
    options: {
      kind: "object",
      children: {
        caseSensitive: { kind: "leaf" },
        leftValue: { kind: "leaf" },
        typeValidation: { kind: "enum", values: new Set(["strict", "loose"]) },
        version: { kind: "leaf" },
      },
    },
    conditions: {
      kind: "array",
      item: {
        kind: "object",
        children: {
          id: { kind: "leaf" },
          leftValue: { kind: "leaf" },
          rightValue: { kind: "leaf" },
          operator: {
            kind: "object",
            children: {
              type: { kind: "enum", values: FILTER_OPERATOR_TYPES },
              // operation's real valid set depends on the sibling "type"
              // value - validated via customValidate on the parent
              // condition object below, not here (a flat per-key enum
              // can't express a cross-field dependency).
              operation: { kind: "leaf" },
              rightType: { kind: "enum", values: FILTER_OPERATOR_TYPES },
              singleValue: { kind: "leaf" },
            },
          },
        },
        customValidate: (value, path, valueIssues) => validateFilterCondition(value, path, valueIssues),
      },
    },
    combinator: { kind: "enum", values: FILTER_COMBINATORS },
  },
};

// "value"'s contents are legitimately free-form (map to whatever the target
// resource's own fields are — a called sub-workflow's declared inputs, a
// sheet's column names, ...), which this offline schema check has no way to
// know and shouldn't guess at, so it's "opaque" (never recursed into,
// regardless of what shape it actually is). "schema" is an array of
// ResourceMapperField, each with its OWN fixed shape, distinct from
// "value"'s free-form contents — missing this caused a real false positive
// on an actual user workflow (not just a synthetic eval case).
//
// mappingMode and each ResourceMapperField's own "type" are typed as loose
// `string` in n8n's own interfaces.ts (ResourceMapperValue.mappingMode:
// string; ResourceMapperField has no closed union for its "type" either,
// since it describes an EXTERNAL system's column/field type, not an n8n
// concept) — no authoritative closed enum exists for either, so both stay
// unchecked ("leaf") rather than risk a false positive on a legitimate value
// n8n itself doesn't constrain.
const RESOURCE_MAPPER_SCHEMA = {
  kind: "object",
  children: {
    mappingMode: { kind: "leaf" },
    value: { kind: "opaque" },
    matchingColumns: { kind: "leaf" },
    schema: {
      kind: "array",
      item: {
        kind: "object",
        children: {
          id: { kind: "leaf" },
          displayName: { kind: "leaf" },
          defaultMatch: { kind: "leaf" },
          canBeUsedToMatch: { kind: "leaf" },
          required: { kind: "leaf" },
          display: { kind: "leaf" },
          type: { kind: "leaf" },
          removed: { kind: "leaf" },
          options: { kind: "opaque" },
          readOnly: { kind: "leaf" },
          defaultValue: { kind: "leaf" },
        },
      },
    },
    attemptToConvertTypes: { kind: "leaf" },
    convertFieldsToString: { kind: "leaf" },
  },
};

// AssignmentValue.type is also a loose `string` in interfaces.ts (no closed
// union) — same reasoning as above, left unchecked.
const ASSIGNMENT_COLLECTION_SCHEMA = {
  kind: "object",
  children: {
    assignments: {
      kind: "array",
      item: {
        kind: "object",
        children: {
          id: { kind: "leaf" },
          name: { kind: "leaf" },
          value: { kind: "opaque" },
          type: { kind: "leaf" },
        },
      },
    },
  },
};

const COMPOSITE_TYPE_SCHEMAS = {
  filter: FILTER_SCHEMA,
  resourceMapper: RESOURCE_MAPPER_SCHEMA,
  assignmentCollection: ASSIGNMENT_COLLECTION_SCHEMA,
};

// resourceLocator's protocol-level wrapper shape
// ({__rl: true, value, mode, cachedResultName, cachedResultUrl}) — "value"
// is opaque (just a plain string/ID). Unlike the composite types above,
// "mode" DOES have a real, statically-knowable enum here — each
// resourceLocator property declares its own `.modes` array (e.g. channelId's
// modes are list/id; documentId's are list/url/id) — so this builds a
// PER-PROPERTY descriptor from the actual property definition instead of a
// single shared constant, giving mode the same precision as everything else.
function describeResourceLocator(prop) {
  const modeNames = Array.isArray(prop.modes) ? prop.modes.map((m) => m.name) : null;
  return {
    kind: "object",
    children: {
      __rl: { kind: "leaf" },
      value: { kind: "opaque" },
      mode: modeNames && modeNames.length ? { kind: "enum", values: new Set(modeNames) } : { kind: "leaf" },
      cachedResultName: { kind: "leaf" },
      cachedResultUrl: { kind: "leaf" },
    },
  };
}

// Dynamic option lists (fetched from an external system at generation/edit
// time via a loadOptionsMethod, e.g. "list all Slack channels" or "list all
// Jira issue types") have no statically-knowable valid-value set — the real
// answer depends on the connected account's actual data, which this offline
// check can't see. Validating against a fixed list here would false-flag
// perfectly legitimate values, so these are left as "leaf" (unchecked)
// rather than "enum".
function hasDynamicOptions(prop) {
  const to = prop.typeOptions;
  return !!(to && (to.loadOptionsMethod || to.loadOptionsDependsOn || to.searchListMethod));
}

// Converts a single n8n INodeProperty into a schema descriptor for its value.
function describeProperty(prop) {
  if (prop.type === "collection" && Array.isArray(prop.options)) {
    return { kind: "object", children: buildChildMap(prop.options) };
  }
  if (prop.type === "fixedCollection" && Array.isArray(prop.options)) {
    const multiple = !!(prop.typeOptions && prop.typeOptions.multipleValues);
    const children = {};
    for (const group of prop.options) {
      const groupSchema = { kind: "object", children: buildChildMap(group.values || []) };
      children[group.name] = multiple ? { kind: "array", item: groupSchema } : groupSchema;
    }
    return { kind: "object", children };
  }
  if (prop.type === "resourceLocator") {
    return describeResourceLocator(prop);
  }
  if (COMPOSITE_TYPE_SCHEMAS[prop.type]) {
    return COMPOSITE_TYPE_SCHEMAS[prop.type];
  }
  if ((prop.type === "options" || prop.type === "multiOptions") &&
      Array.isArray(prop.options) && !hasDynamicOptions(prop)) {
    // .options entries are {name, value, ...} - "value" is what actually
    // ends up in node.parameters, "name" is just the UI display label.
    const values = new Set(prop.options.filter((o) => o && "value" in o).map((o) => o.value));
    return prop.type === "multiOptions" ? { kind: "enumArray", values } : { kind: "enum", values };
  }
  // string/number/boolean/json/dateTime/notice/button/dynamic-options/etc. -
  // a primitive value (possibly an expression string, or a value only
  // knowable from a live external account), no nested keys or fixed value
  // set to check.
  return { kind: "leaf" };
}

// Two DIFFERENT property declarations can legitimately share the same name
// at the same nesting level, gated by displayOptions on some OTHER field
// (e.g. Slack's "channelId" is a plain string for the "create" operation but
// a resourceLocator object for every other operation). Rather than picking
// whichever declaration happens to come last (which would false-flag the
// other operation's valid usage), merge them permissively — but as precisely
// as fairness allows:
//   - same kind: union children / enum value sets.
//   - object vs primitive(leaf/enum): the two shapes are perfectly
//     discriminable at validation time by the ACTUAL value's runtime shape,
//     so a {kind: "variant"} keeps full checking for whichever declaration
//     the value actually matches instead of dropping to unchecked (this is
//     exactly the Slack channelId case — an object value still gets its
//     resourceLocator mode-enum check, a plain string is left alone).
//   - enum vs leaf (both primitive-shaped, NOT discriminable at runtime —
//     a string could legitimately belong to the unconstrained declaration):
//     permissive leaf, since checking the enum would false-flag.
//   - anything else ambiguous (array vs enumArray, etc.): opaque — never
//     false-flag over a shape collision this file's schemas don't produce.
function isPrimitiveKind(d) {
  return d.kind === "leaf" || d.kind === "enum";
}

function mergeDescriptors(a, b) {
  if (!a) return b;
  if (!b) return a;
  if (a.kind === "opaque" || b.kind === "opaque") return { kind: "opaque" };
  if (a.kind !== b.kind) {
    // Runtime-discriminable split: one side an object, the other a primitive.
    const obj = a.kind === "object" ? a : b.kind === "object" ? b : null;
    const other = obj === a ? b : a;
    if (obj) {
      if (other.kind === "variant") {
        return { kind: "variant", objectDesc: mergeDescriptors(obj, other.objectDesc), primitiveDesc: other.primitiveDesc };
      }
      if (isPrimitiveKind(other)) {
        return { kind: "variant", objectDesc: obj, primitiveDesc: other };
      }
    }
    if (a.kind === "variant" || b.kind === "variant") {
      const variant = a.kind === "variant" ? a : b;
      const otherSide = variant === a ? b : a;
      if (isPrimitiveKind(otherSide)) {
        return { kind: "variant", objectDesc: variant.objectDesc, primitiveDesc: mergeDescriptors(variant.primitiveDesc, otherSide) };
      }
      return { kind: "opaque" };
    }
    // enum vs leaf: both primitive-shaped, not discriminable — permissive.
    if (isPrimitiveKind(a) && isPrimitiveKind(b)) return { kind: "leaf" };
    return { kind: "opaque" };
  }
  if (a.kind === "variant") {
    return {
      kind: "variant",
      objectDesc: mergeDescriptors(a.objectDesc, b.objectDesc),
      primitiveDesc: mergeDescriptors(a.primitiveDesc, b.primitiveDesc),
    };
  }
  if (a.kind === "object") {
    const children = Object.assign({}, a.children);
    for (const [k, v] of Object.entries(b.children)) {
      children[k] = mergeDescriptors(children[k], v);
    }
    const merged = { kind: "object", children };
    // Prefer whichever side has a customValidate hook; both sides having one
    // (two DIFFERENT cross-field validators on the same-named property)
    // doesn't happen anywhere in this file's schemas, so first-found is fine.
    if (a.customValidate || b.customValidate) merged.customValidate = a.customValidate || b.customValidate;
    return merged;
  }
  if (a.kind === "array") {
    return { kind: "array", item: mergeDescriptors(a.item, b.item) };
  }
  if (a.kind === "enum" || a.kind === "enumArray") {
    return { kind: a.kind, values: new Set([...a.values, ...b.values]) };
  }
  return a; // both "leaf" - nothing more to merge
}

function buildChildMap(props) {
  const map = {};
  for (const prop of props || []) {
    map[prop.name] = mergeDescriptors(map[prop.name], describeProperty(prop));
  }
  return map;
}

// Walks `value` against `schema` in parallel, appending the full dotted path
// of any key found that isn't declared at that exact location (into
// `issues`), or whose value isn't one of that field's real allowed values
// (into `valueIssues`). Type mismatches (e.g. a string where an object was
// expected, from the same-name/different-shape case above resolving to
// "opaque") are not reported here - they're a different kind of bug than
// "invented key name" or "invented value", and this check's whole purpose is
// those two.
function findMismatches(schema, value, path, issues, valueIssues) {
  if (!schema || schema.kind === "opaque") return;
  if (value === null || value === undefined) return;
  if (schema.kind === "leaf") return;
  if (schema.kind === "variant") {
    // Same-named property declared with different shapes across operations
    // (see mergeDescriptors) — pick the declaration branch matching the
    // value's actual runtime shape; a shape matching neither branch is the
    // "type mismatch" case deliberately not reported by this checker.
    if (typeof value === "object" && !Array.isArray(value)) {
      findMismatches(schema.objectDesc, value, path, issues, valueIssues);
    } else if (!Array.isArray(value)) {
      findMismatches(schema.primitiveDesc, value, path, issues, valueIssues);
    }
    return;
  }
  if (schema.kind === "enum") {
    if (typeof value !== "string" || value.startsWith("=")) return;
    if (!schema.values.has(value)) {
      valueIssues.push({ path, value, validValues: [...schema.values] });
    }
    return;
  }
  if (schema.kind === "enumArray") {
    if (!Array.isArray(value)) return;
    for (const v of value) {
      if (typeof v !== "string" || v.startsWith("=")) continue;
      if (!schema.values.has(v)) valueIssues.push({ path, value: v, validValues: [...schema.values] });
    }
    return;
  }
  if (schema.kind === "array") {
    if (!Array.isArray(value)) return;
    for (const item of value) findMismatches(schema.item, item, path, issues, valueIssues);
    return;
  }
  if (schema.kind === "object") {
    if (typeof value !== "object" || Array.isArray(value)) return;
    if (schema.customValidate) schema.customValidate(value, path, valueIssues);
    for (const [key, childValue] of Object.entries(value)) {
      const childSchema = schema.children[key];
      if (!childSchema) {
        issues.push(path ? `${path}.${key}` : key);
        continue;
      }
      findMismatches(childSchema, childValue, path ? `${path}.${key}` : key, issues, valueIssues);
    }
  }
}

// ---------------------------------------------------------------------------
// Dangling node-name reference check.
//
// n8n expressions reference another node's output by name via one of four
// real syntaxes (verified directly against n8n-workflow's own
// node-reference-parser-utils.ts, used internally for sub-workflow
// extraction/renaming): $('Name'), $("Name"), $node['Name'], $node["Name"],
// $items('Name'), $items("Name"). ($node.Name dot-notation is deliberately
// NOT checked here - it's JS-identifier-constrained and rare for n8n's
// typically human-readable, space-containing node names, and safely
// distinguishing it from unrelated property-chain text is much harder to do
// without false positives.)
//
// Expressions are only evaluated by n8n in string values that start with
// "=" - EXCEPT a Code node's "jsCode", which is raw JavaScript that can
// reference other nodes the exact same way without any "=" prefix (verified
// in the same source file's applyParameterMapping: it treats a value as
// expression-bearing when `charAt(0) === '=' || keyOfValue === 'jsCode'`).
// ---------------------------------------------------------------------------
const NODE_REF_PATTERNS = [
  /\$\(\s*(['"])((?:(?!\1)[\s\S])*)\1\s*\)/g,
  /\$node\[\s*(['"])((?:(?!\1)[\s\S])*)\1\s*\]/g,
  /\$items\(\s*(['"])((?:(?!\1)[\s\S])*)\1/g,
];

function collectNodeNameReferences(value, key, refs) {
  if (typeof value === "string") {
    if (value.startsWith("=") || key === "jsCode") {
      for (const pattern of NODE_REF_PATTERNS) {
        pattern.lastIndex = 0;
        let match;
        while ((match = pattern.exec(value))) refs.add(match[2]);
      }
    }
    return;
  }
  if (Array.isArray(value)) {
    for (const item of value) collectNodeNameReferences(item, key, refs);
    return;
  }
  if (value && typeof value === "object") {
    for (const [k, v] of Object.entries(value)) collectNodeNameReferences(v, k, refs);
  }
}

const instanceCache = new Map();

function loadInstance(type) {
  if (instanceCache.has(type)) return instanceCache.get(type);
  const entry = NODE_TYPE_MAP[type];
  if (!entry) {
    instanceCache.set(type, null);
    return null;
  }
  const [modulePath, exportName] = entry;
  const mod = require(modulePath);
  const inst = new mod[exportName]();
  instanceCache.set(type, inst);
  return inst;
}

function getDescriptionForVersion(inst, version) {
  if (inst.nodeVersions) {
    const versioned = inst.nodeVersions[version] || inst.nodeVersions[inst.currentVersion];
    return versioned.description;
  }
  return inst.description;
}

function checkNode(node, allNodeNames) {
  const finding = { node: node.name, type: node.type, unknownParams: [], invalidValues: [], danglingNodeReferences: [] };

  const inst = loadInstance(node.type);
  if (inst) {
    const desc = getDescriptionForVersion(inst, node.typeVersion);
    const topLevelChildren = buildChildMap(desc.properties);
    if (desc.polling) {
      for (const [k, v] of Object.entries(buildChildMap(commonPollingParameters))) {
        topLevelChildren[k] = mergeDescriptors(topLevelChildren[k], v);
      }
    }
    const topLevelSchema = { kind: "object", children: topLevelChildren };
    const issues = [];
    const valueIssues = [];
    findMismatches(topLevelSchema, node.parameters, "", issues, valueIssues);
    // Report just the leaf key name (matching prior output format), not the
    // full dotted path — downstream consumers (the judge/report) just display
    // these as a flat list of unknown-param names.
    finding.unknownParams = [...new Set(issues.map((p) => p.split(".").pop()))];
    finding.invalidValues = valueIssues;
  }
  // Dangling-reference check doesn't need a recognized schema at all - it's
  // pure string-content parsing, so it runs even for unrecognized node types.
  const refs = new Set();
  collectNodeNameReferences(node.parameters, null, refs);
  finding.danglingNodeReferences = [...refs].filter((name) => !allNodeNames.has(name));

  const hasAnyFinding = finding.unknownParams.length || finding.invalidValues.length || finding.danglingNodeReferences.length;
  return hasAnyFinding ? finding : null;
}

// A missing n8n-workflow/n8n-nodes-base/n8n-core install fails EVERY node
// with the same MODULE_NOT_FOUND — that's one setup problem, not N separate
// per-node problems. Detected once and reported as a single setupError so
// the caller can print one message and stop invoking this script for the
// rest of the run, instead of the same warning repeating per node per turn.
function isMissingOwnDependency(e) {
  return e && e.code === "MODULE_NOT_FOUND" &&
    /n8n-workflow|n8n-nodes-base|n8n-core/.test(String(e.message || ""));
}

// Flags (a) node types claiming the "n8n-nodes-base." prefix that don't
// exist in the installed package at all, and (b) typeVersions that aren't
// in the node's real version list. (b)'s one honest caveat: if the locally
// installed n8n-nodes-base is OLDER than the n8n instance the workflow will
// run on, a legitimately newer typeVersion would be wrongly flagged — the
// message names the installed package version so that's diagnosable.
// package.json pins "latest" precisely to keep that window small.
function checkNodeTypeExists(node) {
  if (!knownBaseNodes || typeof node.type !== "string") return null;
  if (!node.type.startsWith("n8n-nodes-base.")) return null; // unverifiable prefix — skip
  const shortName = node.type.slice("n8n-nodes-base.".length);
  if (!knownBaseNodes.has(shortName)) {
    return { kind: "unknownType", node: node.name, type: node.type };
  }
  if (node.typeVersion !== undefined) {
    const versions = knownBaseNodes.get(shortName);
    if (!versions.some((v) => Number(v) === Number(node.typeVersion))) {
      return {
        kind: "unknownVersion", node: node.name, type: node.type,
        typeVersion: node.typeVersion, knownVersions: versions,
        installedPackage: `n8n-nodes-base@${installedBaseVersion}`,
      };
    }
  }
  return null;
}

function main() {
  let raw = "";
  process.stdin.on("data", (chunk) => (raw += chunk));
  process.stdin.on("end", () => {
    const issues = [];
    const unknownNodeTypes = [];
    const unknownTypeVersions = [];
    const warnings = [];
    try {
      const workflow = JSON.parse(raw);
      const allNodeNames = new Set((workflow.nodes || []).map((n) => n && n.name).filter(Boolean));
      for (const node of workflow.nodes || []) {
        try {
          const typeFinding = checkNodeTypeExists(node);
          if (typeFinding && typeFinding.kind === "unknownType") {
            unknownNodeTypes.push(typeFinding);
          } else if (typeFinding) {
            unknownTypeVersions.push(typeFinding);
          }
          const finding = checkNode(node, allNodeNames);
          if (finding) issues.push(finding);
        } catch (e) {
          if (isMissingOwnDependency(e)) {
            process.stdout.write(JSON.stringify({
              issues: [],
              warnings: [],
              setupError: "npm packages not installed — copy check_params.js and package.json to "
                + "local scratch space (e.g. /tmp/n8n_schema_check_cache/) and run "
                + "'npm install --ignore-scripts' there. Don't install into this file's own "
                + "directory if it's inside a git-synced Workspace folder — node_modules' symlinks "
                + "(e.g. node_modules/.bin/*) can break Databricks Repos' git-status UI, and a "
                + "network-backed /Workspace filesystem makes every cold require() much slower "
                + "than local disk. (" + String((e && e.message) || e) + ")",
            }));
            return;
          }
          // A node's schema failing to load/resolve for some OTHER reason
          // (not our own missing deps) is a real, node-specific problem —
          // surface it instead of swallowing it, but keep checking the rest.
          warnings.push(`${node.name} (${node.type}): ${String((e && e.message) || e)}`);
        }
      }
      // Split the combined per-node findings back into three separate
      // top-level arrays, one per check category, so each can become its
      // own independent pass/fail check on the Python side (mirroring how
      // "unknownParams" alone used to be the entirety of "issues").
      const unknownParamIssues = issues
        .filter((f) => f.unknownParams.length)
        .map((f) => ({ node: f.node, type: f.type, unknownParams: f.unknownParams }));
      const invalidValueIssues = issues
        .filter((f) => f.invalidValues.length)
        .map((f) => ({ node: f.node, type: f.type, invalidValues: f.invalidValues }));
      const danglingRefIssues = issues
        .filter((f) => f.danglingNodeReferences.length)
        .map((f) => ({ node: f.node, type: f.type, danglingNodeReferences: f.danglingNodeReferences }));
      process.stdout.write(JSON.stringify({
        issues: unknownParamIssues,
        invalidValues: invalidValueIssues,
        danglingNodeReferences: danglingRefIssues,
        unknownNodeTypes,
        unknownTypeVersions,
        warnings,
      }));
    } catch (e) {
      process.stdout.write(JSON.stringify({ issues: [], invalidValues: [], danglingNodeReferences: [], unknownNodeTypes: [], unknownTypeVersions: [], warnings: [], error: String((e && e.message) || e) }));
    }
  });
}

main();
