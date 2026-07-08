#!/usr/bin/env node
/**
 * Reads a workflow JSON on stdin, and for every node whose type is one we
 * have a local n8n-nodes-base definition for, flags any parameter key that
 * doesn't exist anywhere in that node type's own declared property schema.
 *
 * This exists because neither n8n's own NodeHelpers.getNodeParametersIssues
 * nor workflow activation catch invented/hallucinated parameter names (both
 * verified empirically) — they only catch missing *required* fields. Since
 * a fabricated parameter is silently ignored at runtime rather than erroring,
 * the only way to catch it is comparing against the node's real schema
 * directly, which is what this does.
 *
 * Only the node types actually used by the Workflow Builder are covered —
 * unrecognized types (e.g. Ibotta's own custom KA node) are skipped, not
 * flagged as errors, since we have no schema to compare against.
 *
 * Output (stdout, always valid JSON): {"issues": [{"node", "type", "unknownParams"}]}
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
};

// Protocol-level wrapper keys used by n8n's resourceLocator field shape
// ({__rl: true, value, mode, cachedResultName, ...}) — not user-declared
// schema properties, so never flag these as unknown. Recursing into them is
// harmless here since they're primitives, unlike OPAQUE_KEYS below.
const WRAPPER_KEYS = new Set(["__rl", "mode", "cachedResultName", "cachedResultUrl"]);

// Keys whose contents are legitimately free-form and must never be checked
// at all, not even recursed into — resourceMapper's "value" object maps to
// whatever the target resource's own fields are (a called sub-workflow's
// declared inputs, a sheet's column names, ...), which this offline schema
// check has no way to know and shouldn't guess at. (Also covers
// resourceLocator's "value", which is just a plain string/ID — recursing
// into a primitive is already a no-op, so folding it in here too is safe.)
const OPAQUE_KEYS = new Set(["value"]);

// n8n's shared composite parameter types (filter, resourceMapper,
// assignmentCollection) have a FIXED runtime value shape defined by n8n's
// own TypeScript interfaces (FilterValue, FilterConditionValue,
// FilterOperatorValue, FilterOptionsValue, ResourceMapperValue,
// AssignmentCollectionValue, AssignmentValue — verified directly against
// n8n-workflow's interfaces.d.ts), not declared via the property's own
// .options array the way collection/fixedCollection are. Treating "unknown
// to our .options recursion" as "invented" for these was a real bug: it
// flagged genuine, standard fields like leftValue/operator/combinator (real
// filter-condition fields) and mappingMode/value (real resourceMapper
// fields — mappingMode/value is literally n8n's own default value shape
// for it) as hallucinated.
const COMPOSITE_TYPE_KEYS = {
  filter: ["options", "conditions", "combinator", "caseSensitive", "leftValue",
           "typeValidation", "version", "id", "operator", "rightValue",
           "type", "operation", "rightType", "singleValue"],
  resourceMapper: ["mappingMode", "value", "matchingColumns", "schema",
                   "attemptToConvertTypes", "convertFieldsToString"],
  assignmentCollection: ["assignments", "id", "name", "value", "type"],
};

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

function collectDeclaredNames(properties, set = new Set()) {
  for (const prop of properties || []) {
    set.add(prop.name);
    if (prop.type === "collection" && Array.isArray(prop.options)) {
      collectDeclaredNames(prop.options, set);
    } else if (prop.type === "fixedCollection" && Array.isArray(prop.options)) {
      for (const opt of prop.options) {
        set.add(opt.name);
        if (Array.isArray(opt.values)) collectDeclaredNames(opt.values, set);
      }
    } else if (COMPOSITE_TYPE_KEYS[prop.type]) {
      for (const k of COMPOSITE_TYPE_KEYS[prop.type]) set.add(k);
    }
  }
  return set;
}

function collectUsedKeys(obj, set = new Set()) {
  if (obj === null || typeof obj !== "object") return set;
  if (Array.isArray(obj)) {
    for (const item of obj) collectUsedKeys(item, set);
    return set;
  }
  for (const [k, v] of Object.entries(obj)) {
    if (OPAQUE_KEYS.has(k)) continue; // free-form contents — don't add, don't recurse
    if (!WRAPPER_KEYS.has(k)) set.add(k);
    collectUsedKeys(v, set);
  }
  return set;
}

function checkNode(node) {
  const inst = loadInstance(node.type);
  if (!inst) return null; // unrecognized type — not our concern, skip silently
  const desc = getDescriptionForVersion(inst, node.typeVersion);
  const declared = collectDeclaredNames(desc.properties);
  if (desc.polling) collectDeclaredNames(commonPollingParameters, declared);
  const used = collectUsedKeys(node.parameters);
  const unknown = [...used].filter((k) => !declared.has(k));
  return unknown.length ? { node: node.name, type: node.type, unknownParams: unknown } : null;
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

function main() {
  let raw = "";
  process.stdin.on("data", (chunk) => (raw += chunk));
  process.stdin.on("end", () => {
    const issues = [];
    const warnings = [];
    try {
      const workflow = JSON.parse(raw);
      for (const node of workflow.nodes || []) {
        try {
          const finding = checkNode(node);
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
      process.stdout.write(JSON.stringify({ issues, warnings }));
    } catch (e) {
      process.stdout.write(JSON.stringify({ issues: [], warnings: [], error: String((e && e.message) || e) }));
    }
  });
}

main();
