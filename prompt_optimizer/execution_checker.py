"""
Pre-deployment execution-trace check for the custom-RAG pipeline.

Distinct from judge.py's Layer 3 soundness reviewer, which only runs AFTER
generation, for benchmarking/measurement — its findings never feed back into
what the model actually returns. This module is the same idea wired INTO the
generation loop itself: a cheap, narrowly-scoped second opinion that runs
before a structurally-valid response is accepted, so the specific failure
mode it targets gets a chance to be fixed before the workflow is ever
returned, not just scored afterward.

Scoped to exactly the "structurally-shaped-but-wouldn't-actually-execute-
correctly" pattern found to underlie nearly every real (non-reviewer-false-
positive) blocker in a full review of the hard-scenario benchmark traces —
NOT the broader "is this generally sound" question Layer 3 asks (naming,
scalability, style nits). Three checks, matching the self-check instruction
added to the Workflow Builder prompt itself:
  1. Does every $('Node')/$json cross-reference resolve given the node's
     actual position in the graph and every path that reaches it?
  2. Does every approval gate / self-loop guard cover EVERY action it's
     meant to gate, not just one of several, and sit upstream of the
     mutation it's supposed to prevent?
  3. Does every multi-branch convergence that needs synchronized data
     actually use a Merge node (a plain node with multiple incoming
     connections runs once per branch, not once with all combined)?

Modeled on relevance_filter.py's shape: cheap model (Haiku/fast_generation_
endpoint, not the reasoning-heavy generation model), narrow JSON-only output,
fails open — a broken checker must never make a workflow worse than not
running the checker at all.
"""
import json
from dataclasses import dataclass
from typing import List

import httpx
from .llm_response import content_to_text

_EXECUTION_CHECK_SYSTEM = """\
You are tracing through an n8n workflow's actual execution — not reviewing \
general quality. You will see a workflow's JSON. Check ONLY these three \
things:

1. CROSS-NODE REFERENCES: for every expression referencing another node's \
output ($('Node Name').item.json.field, or $json fields assumed to have \
passed through from an earlier node), will that node actually have run and \
produced that field by the time this expression evaluates, in EVERY path \
that reaches it — not just the path the workflow's author had in mind?

2. GUARD/GATE COVERAGE: for every approval gate or self-loop/anti-recursion \
guard, does it sit upstream of EVERY action it's meant to gate (not just \
one of several outbound sends), and before — not after — the mutation it's \
supposed to prevent?

3. MERGE/SYNCHRONIZATION: for every point where multiple branches need to \
be present TOGETHER (e.g. deduplicating across parallel fetches), is there \
an actual Merge node with numberOfInputs matching the branch count? A plain \
node (Code, Set, etc.) with multiple incoming connections runs once PER \
BRANCH, not once with all of them combined — flag this if the workflow's \
logic assumes otherwise.

Do NOT comment on anything else — no naming, no style, no scalability, no \
missing error handling, no credential concerns. If a category has no issue, \
say nothing about it. If you are not confident something is actually wrong, \
leave it out — a false positive here sends the model chasing a problem \
that doesn't exist.

VERIFIED PLATFORM FACTS — these override your own priors about n8n; every \
one of them was previously flagged (wrongly) by a checker like you, and each \
wrong flag sent the builder off breaking a correct workflow:
- With a Structured Output Parser attached, an AI Agent's output item is \
ALWAYS { "output": <parsed object> } — exactly ONE level of wrapping. So \
$json.output.field and $('AI Agent').item.json.output.field are CORRECT and \
COMPLETE references to a schema field. There is NO second/double wrapping: \
never "correct" these to $json.output.output.field — that path does not \
exist and demanding it breaks a working workflow.
- The approval-gate DM goes to the WORKFLOW OWNER, so a hardcoded owner \
user ID in "Get DM Channel ID" is CORRECT and intended. Never demand that \
it be derived dynamically from the trigger/chat input or "routed to the \
right person" — the approver is the owner, not the person who triggered the \
run or the subject of the message.
- $('Node Name') references resolve from executed-node data regardless of \
how many nodes sit between the referenced node and the current one — \
"the data context may not be preserved" is not a real n8n failure mode. \
Only flag a reference when the referenced node may genuinely NOT HAVE \
EXECUTED on a path that reaches the expression (mutually exclusive branch, \
error-only branch).
- A node with "onError": "continueErrorOutput" has a SECOND output (index \
1) that fires ONLY when the node errors. A connection from output index 1 \
is an error branch, not a parallel always-on branch — do not claim it "runs \
every time".
- Nodes on a branch that never executes never evaluate their expressions — \
an expression on an unreached node cannot fail at runtime. Do not flag \
"if this branch doesn't run, its expressions will fail".
- This platform applies an approval gate (Execute Workflow call to the \
approval sub-workflow) upstream of outbound Slack/email sends BY DEFAULT, \
including error alerts. The gate running on a send is intended behavior — \
never suggest limiting or removing it UNLESS the user's request explicitly \
asked for that specific send to be automatic/no-approval, in which case the \
ungated send is also intended and not an issue.
- Split In Batches' "done" output (index 0) accumulates ALL items fed back \
into the loop across every iteration (live-verified) — aggregating from the \
done output after a loop is correct; never flag it as "only the last batch".
- A Merge fed by mutually exclusive branches does NOT deadlock \
(live-verified): "append" mode emits whatever arrived and the workflow \
proceeds. The one real problem shape: "combine" mode emits ZERO items when \
one input never ran, silently skipping everything downstream — flag ONLY \
that (fix: append mode or combine's include-unpaired option).
- The Jira transitions lookup returns one item per transition with flat \
id/name fields (live-verified) — iterating $input.all() on it is correct.
- The user's original request is provided for context: something the user \
explicitly asked for is a requirement, not an issue.

Return ONLY valid JSON: {"issues": ["<node name>: <specific problem and \
why it would actually fail at runtime>", ...]} — empty list if none of the \
three categories have a real issue.
"""


@dataclass
class ExecutionCheckResult:
    issues: List[str]


async def check_execution(
    client: httpx.AsyncClient,
    endpoint_url: str,
    headers: dict,
    workflow_json_text: str,
    user_request: str = "",
) -> List[str]:
    """
    Returns a list of issue strings (empty if none found, or if the check
    itself failed for any reason — fails open, matching relevance_filter.py:
    a broken check must degrade to "no check ran," never to "block every
    workflow" or crash the generation loop it's embedded in.

    user_request gives the checker the request the workflow is FOR. Added
    after a live benchmark showed the blind checker flagging intentional,
    user-required behavior as bugs (e.g. "the approval workflow will run on
    every review" — which is exactly what the platform mandates).
    """
    user_content = f"Workflow JSON:\n{workflow_json_text}"
    if user_request.strip():
        user_content = (
            f"The user's original request this workflow implements:\n"
            f"{user_request}\n\n{user_content}"
        )
    try:
        resp = await client.post(
            endpoint_url,
            headers=headers,
            json={
                "messages": [
                    {"role": "system", "content": _EXECUTION_CHECK_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": 800,
                "temperature": 0.0,
            },
            timeout=60,
        )
        resp.raise_for_status()
        body = resp.json()
        choices = body.get("choices") or []
        content = content_to_text((choices[0].get("message") or {}).get("content")) if choices else None
        if not content:
            return []
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end == -1 or end < start:
            return []
        parsed = json.loads(content[start:end + 1])
        issues = parsed.get("issues", [])
        return [str(i) for i in issues if isinstance(i, str) and i.strip()]
    except Exception:
        return []
