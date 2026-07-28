"""
Layer 4 of the eval upgrade — a small, hand-crafted set of deliberately hard
scenarios, as opposed to the ~200 LLM-generated trigger×output combinations in
synthetic_data.py (which are effectively "does the KA know this integration
exists" hallucination tests).

Motivation: a KA can score ~99% on the synthetic dataset while still failing
almost every real multi-step request, because individually-correct params/
enums don't make a workflow's *graph* correct — see validator.py's Layer 1/2
checks and judge.py's soundness review (Layer 3), which these scenarios are
designed to exercise. Each scenario below is modeled on a REAL observed
failure mode (self-triggering Slack loops, unwired Structured Output Parsers,
disconnected nodes, orphaned branches) rather than invented edge cases.

Deliberately NOT LLM-generated: hand-crafting keeps the specific hard part of
each scenario precisely engineered and reviewable, instead of leaving it to
chance the way synthetic_data.py's generation prompts do.

`expected_behavior` here doubles as a rubric for the Layer 3 judge — it spells
out the specific hard requirement(s) a correct workflow must satisfy, not just
a vague restatement of the ask.
"""
from .synthetic_data import SyntheticInput

HARD_SCENARIOS: list = [
    SyntheticInput(
        text=(
            "We have a #support-intake Slack channel. Whenever anyone posts a message "
            "there, or replies in a thread, I want a bot to look at the FULL thread "
            "history (not just the newest message), send it to an AI agent that "
            "classifies it as a bug report, feature request, or question and writes a "
            "one-sentence summary, then posts that classification as a threaded reply. "
            "It also needs to log every classified thread as a new row in our "
            "'Support Intake Log' spreadsheet — columns are exactly 'Timestamp', "
            "'Thread Link', 'Category', 'Summary', and 'Requester'. Obviously the bot's "
            "own replies must never re-trigger itself and start an infinite loop."
        ),
        category="hard_slack_intake_self_loop",
        trigger="slack",
        outputs=["slack_message", "sheets_update"],
        has_approval=False,
        is_ood=False,
        expected_behavior=(
            "Slack Trigger on the channel (message/app_mention). Before doing anything "
            "else, the incoming event's user/bot_id must be checked against the bot's "
            "own identity (e.g. subtype != bot_message, or user !== the bot's user id) "
            "so its own replies never re-enter the chain. On every valid execution "
            "(including thread replies) it must unconditionally fetch the FULL thread "
            "via a Slack 'get replies' call keyed on thread_ts — not just the triggering "
            "message — and feed the whole reconstructed conversation to an AI Agent. The "
            "AI Agent must have a Structured Output Parser wired via the ai_outputParser "
            "connection type (not main) constraining output to at least "
            "{category, summary}. Results fan out to two independent, both-reachable "
            "actions: a threaded Slack reply (same thread_ts) AND a Google Sheets append "
            "row mapping Timestamp/Thread Link/Category/Summary/Requester to the correct "
            "columns/params — this is a second output, not a replacement for the Slack "
            "reply."
        ),
    ),
    SyntheticInput(
        text=(
            "Build an AI agent that can look up open Jira tickets for a project, hit an "
            "internal HTTP API to check deployment status, and pull rows from a "
            "'Runbook' Google Sheet, then decide what to tell an on-call engineer. Give "
            "it short-term memory of the conversation and make it return a structured "
            "{ recommendation, confidence, sources } object."
        ),
        category="hard_ai_agent_multi_tool_memory",
        trigger="slack",
        outputs=["slack_message"],
        has_approval=False,
        is_ood=False,
        expected_behavior=(
            "One AI Agent node with FIVE distinct sub-nodes, each wired via its correct "
            "connection type (never main): a chat model (ai_languageModel), a memory "
            "node such as window buffer memory (ai_memory), and three tools (ai_tool) — "
            "a Jira tool, an HTTP Request tool, and a Google Sheets tool — plus a "
            "Structured Output Parser (ai_outputParser) constraining the final answer to "
            "{recommendation, confidence, sources}. All five sub-nodes must actually "
            "target the AI Agent node, not be left dangling or wired to each other."
        ),
    ),
    SyntheticInput(
        text=(
            "New emails to our support inbox should get AI-classified into billing, "
            "technical, or general. Technical and general replies can go out "
            "automatically, but ANY billing-related reply needs a human to approve it "
            "in Slack first. Regardless of category, log every email (subject, category, "
            "whether it was auto-sent or approved) to one shared spreadsheet."
        ),
        category="hard_switch_merge_approval",
        trigger="gmail",
        outputs=["email", "sheets_update"],
        has_approval=True,
        is_ood=False,
        expected_behavior=(
            "Gmail Trigger -> AI Agent classifies into 3 categories -> a 3-output Switch. "
            "Only the billing branch routes through a Slack approval gate (message with "
            "Approve/Deny, waiting on a subsequent Slack Trigger/response) before sending; "
            "technical and general branches send directly with no approval step — an "
            "approval gate hard-wired onto ALL three branches, or onto none, is wrong. "
            "All three branches, after their respective send (or deny/no-send) paths, "
            "must reconverge on a single shared Google Sheets logging step (e.g. via a "
            "Merge node) rather than three independent, duplicated logging nodes. Every "
            "branch — including the deny path — must be reachable; none should dead-end."
        ),
    ),
    SyntheticInput(
        text=(
            "When a Jira ticket in our PLAT project is updated, have an AI agent review "
            "it and, if it looks like a P1, add a triage comment tagging @oncall and "
            "bump priority. We also want the trigger to catch new comments so nothing "
            "slips through — but obviously it can't react to its OWN triage comment and "
            "keep commenting forever."
        ),
        category="hard_jira_comment_self_loop",
        trigger="jira",
        outputs=["docs_update"],
        has_approval=False,
        is_ood=False,
        expected_behavior=(
            "Jira Trigger configured to fire on issue updates and comments. Before the AI "
            "assessment runs, the workflow must guard against reacting to its own "
            "previously-added comment — e.g. checking the triggering comment/update's "
            "author against the bot's own Jira account, or restricting the trigger's "
            "watched fields/events so the bot's own comment-creation call doesn't "
            "recirculate. Without this guard, every bot-authored comment would re-fire "
            "the same trigger and comment again indefinitely."
        ),
    ),
    SyntheticInput(
        text=(
            "Whenever a new file lands in our 'Weekly Reports' Drive folder, pull the "
            "matching Google Doc's content, have AI summarize it into 3 bullet points "
            "plus a 1-10 risk score, append that to our 'Report Tracker' sheet, and post "
            "a Slack message with the doc link and the risk score."
        ),
        category="hard_drive_docs_sheets_chain",
        trigger="drive",
        outputs=["sheets_update", "slack_message"],
        has_approval=False,
        is_ood=False,
        expected_behavior=(
            "Google Drive Trigger (folderToWatch pointed at the folder) -> Google Docs "
            "get/read of the new file's content -> AI Agent with a Structured Output "
            "Parser returning {bullets, risk_score} -> Google Sheets append row -> Slack "
            "message including both the doc link and risk score. Five sequential hops "
            "across three different Google node families, each needing its own correct "
            "resourceLocator shape ({__rl, mode, value}) — a common failure is reusing "
            "one node family's field/resource shape on another."
        ),
    ),
    SyntheticInput(
        text=(
            "When a card is created in our 'Intake' Trello list, have AI look at the "
            "card and fill in a priority label and a one-line description update on that "
            "same card. We also watch that list for any card updates, so make sure the "
            "bot updating its own card doesn't cause it to process the card over and over."
        ),
        category="hard_trello_card_self_update",
        trigger="trello",
        outputs=["trello_card"],
        has_approval=False,
        is_ood=False,
        expected_behavior=(
            "Trello Trigger subscribed to the list (fires on every webhook event for "
            "that model, including updateCard — there is no event-type filter on this "
            "node). Because the bot's own card update will re-fire the same trigger, the "
            "workflow must include a guard before re-processing — e.g. checking whether "
            "the priority label/description already match what the bot would set, or "
            "checking the update's originating member — so it does not loop indefinitely "
            "re-labeling the same card."
        ),
    ),
    SyntheticInput(
        text=(
            "Every morning at 9am, pull the last 24 hours of messages from #eng-updates, "
            "#product-updates, and #support-escalations, dedupe anything mentioned in "
            "more than one channel, and have AI write ONE combined digest that gets "
            "posted to #daily-digest."
        ),
        category="hard_cron_multi_channel_digest",
        trigger="cron",
        outputs=["slack_message"],
        has_approval=False,
        is_ood=False,
        expected_behavior=(
            "Cron Trigger (daily 9am) fans out to three independent Slack 'get channel "
            "history' calls (one per channel, each scoped to the last 24h), which must "
            "reconverge — e.g. via a Merge node — into a SINGLE AI Agent call that "
            "dedupes and writes one digest, rather than three separate AI Agent calls or "
            "three separate Slack posts. Only one final message should reach "
            "#daily-digest."
        ),
    ),
    SyntheticInput(
        text=(
            "New emails to support@ should get an AI-drafted reply, but nothing goes out "
            "without a human approving it in Slack first. If approved, send the reply. "
            "If denied, just log it to a sheet and stop — don't send anything."
        ),
        category="hard_gmail_approval_reply",
        trigger="gmail",
        outputs=["email", "sheets_update"],
        has_approval=True,
        is_ood=False,
        expected_behavior=(
            "Gmail Trigger -> AI Agent drafts a reply (Structured Output Parser for the "
            "reply body/subject) -> Slack approval message (Approve/Deny). Because "
            "waiting for the human's response is itself a second Slack Trigger, that "
            "trigger needs its own self-loop guard (it must not react to the bot's own "
            "approval-request message). The approve path must lead only to the Gmail "
            "send node; the deny path must lead only to a Sheets log node — each branch "
            "reachable and doing ONLY its own action, not both branches converging back "
            "onto the send step."
        ),
    ),
    SyntheticInput(
        text=(
            "When a new event is created on our shared team calendar, check if it "
            "overlaps with any other existing event in that same time window. If there's "
            "a conflict, alert the organizer in Slack. If there's no conflict, don't do "
            "anything else — no need to notify anyone."
        ),
        category="hard_calendar_conflict_check",
        trigger="calendar",
        outputs=["slack_message"],
        has_approval=False,
        is_ood=False,
        expected_behavior=(
            "Google Calendar Trigger (event created) -> a lookup of existing events in "
            "the same window (e.g. Google Calendar 'get many' scoped to that time range) "
            "-> AI Agent or IF node decides conflict yes/no -> only the conflict branch "
            "sends a Slack alert. The no-conflict branch is a legitimate dead end (a "
            "genuine no-op, e.g. a NoOp node or simply an unconnected IF output) and must "
            "still be a valid, intentional branch of the graph rather than a sign the "
            "workflow is incomplete."
        ),
    ),
    SyntheticInput(
        text=(
            "We want a Slack bot that only responds when someone @-mentions it in "
            "#ask-data-team, answering questions about our data pipelines. It needs to "
            "remember earlier turns in the SAME thread, including its own past answers, "
            "so people can ask follow-ups — but it obviously shouldn't respond to its own "
            "messages."
        ),
        category="hard_slack_thread_context_bot",
        trigger="slack",
        outputs=["slack_message"],
        has_approval=False,
        is_ood=False,
        expected_behavior=(
            "Slack Trigger scoped to app_mention events -> self-message guard (bot_id/"
            "subtype check) before proceeding -> full-thread fetch via 'get replies' on "
            "every execution, which necessarily includes the bot's OWN prior replies in "
            "that thread as conversational context for the AI Agent. The key nuance: the "
            "self-loop guard only blocks the CURRENT triggering event from being the "
            "bot's own message — it must NOT strip the bot's past messages out of the "
            "thread history fed to the AI, since those are needed for multi-turn context."
        ),
    ),
    SyntheticInput(
        text=(
            "We get JSON form submissions posted to a webhook. If the payload has all "
            "required fields (name, email, message) and they're non-empty, add a row to "
            "our leads sheet. If anything's missing or empty, post the raw payload to "
            "#ops so someone can look at it."
        ),
        category="hard_webhook_validation_branch",
        trigger="cron",
        outputs=["sheets_update", "slack_message"],
        has_approval=False,
        is_ood=False,
        expected_behavior=(
            "Webhook node -> an IF node validating name/email/message are all present "
            "and non-empty. BOTH outputs must be wired to genuinely different actions: "
            "true -> Google Sheets append; false -> Slack message to #ops with the raw "
            "payload. A common failure is only wiring the true branch and leaving the "
            "false output disconnected (or vice versa) — both must be reachable and do "
            "distinct things."
        ),
    ),
    SyntheticInput(
        text=(
            "Have an AI agent review incoming vendor contracts (pasted text) and decide "
            "{decision: approve/reject/escalate, reason}. If for some reason the AI's "
            "output doesn't come back in the right structure, don't just fail silently — "
            "alert an engineer in Slack so someone knows to look at it."
        ),
        category="hard_output_parser_fallback",
        trigger="cron",
        outputs=["slack_message"],
        has_approval=False,
        is_ood=False,
        expected_behavior=(
            "AI Agent with a Structured Output Parser constraining output to "
            "{decision, reason}, PLUS explicit handling for the parser/agent failing "
            "(e.g. onError: continue with an error output, or a dedicated error branch) "
            "that routes to a Slack alert to an engineer — rather than the workflow just "
            "stopping with no visible failure path when structured parsing fails."
        ),
    ),
    SyntheticInput(
        text=(
            "We have a Google Sheet with hundreds of support ticket rows that need AI "
            "classification. Process them in batches so we don't hit rate limits, but "
            "only send ONE Slack summary message at the end with total counts per "
            "category — not one message per row."
        ),
        category="hard_batch_sheet_classification",
        trigger="sheets",
        outputs=["slack_message"],
        has_approval=False,
        is_ood=False,
        expected_behavior=(
            "Google Sheets read -> Split In Batches (Loop Over Items) -> per-item AI "
            "classification inside the loop, with the loop-back edge correctly wired "
            "back to the batch node's 'loop' output (not its 'done' output). Aggregation "
            "of per-category counts and the single summary Slack message must happen "
            "AFTER the loop — connected from the batch node's 'done' output — not inside "
            "the loop body, which is the common failure that produces one message per "
            "row instead of one total."
        ),
    ),
    SyntheticInput(
        text=(
            "Build an AI agent for engineers that can both create a Jira ticket AND kick "
            "off our existing 'Send Slack Approval' sub-workflow as tools it can call, "
            "so it can ask a human before creating high-priority tickets."
        ),
        category="hard_subworkflow_tool_reuse",
        trigger="slack",
        outputs=["slack_message"],
        has_approval=True,
        is_ood=False,
        expected_behavior=(
            "AI Agent with two tools wired via ai_tool: a Jira 'create issue' tool, and "
            "an Execute Workflow Tool node pointing at the existing 'Send Slack Approval' "
            "sub-workflow (referenced as an opaque placeholder ID/name per platform "
            "convention — never fabricated with invented specifics). Both tools must "
            "target the SAME AI Agent node; a common failure is wiring one tool to the "
            "agent and leaving the second tool floating or wired to the wrong node."
        ),
    ),
    SyntheticInput(
        text=(
            "I want the same 'new lead' automation (log to sheet, notify #sales) to kick "
            "off whether someone submits our webhook form OR nothing's come in for a "
            "full day and the cron catch-all runs a sweep — either one should lead to "
            "the same downstream steps."
        ),
        category="hard_multi_trigger_or_logic",
        trigger="cron",
        outputs=["sheets_update", "slack_message"],
        has_approval=False,
        is_ood=False,
        expected_behavior=(
            "TWO separate trigger nodes (Webhook and Cron) both feeding into the SAME "
            "downstream chain (Sheets log -> Slack notify), rather than duplicating the "
            "downstream logic per trigger or only wiring one of the two triggers. "
            "Reachability must be judged from EITHER valid entry point, not assume a "
            "single trigger."
        ),
    ),
    SyntheticInput(
        text=(
            "Every Monday, pull last week's numbers from our 'Weekly Metrics' sheet, "
            "have AI turn them into a few slides, and before anything gets shared, post "
            "a Slack approval message to my manager. If they approve, post the deck link "
            "to #leadership. If they deny, do nothing further."
        ),
        category="hard_slides_generation_approval",
        trigger="cron",
        outputs=["slides_create", "slack_message"],
        has_approval=True,
        is_ood=False,
        expected_behavior=(
            "Cron Trigger (weekly) -> Sheets read -> AI Agent turns data into slide "
            "content -> Google Slides create -> Slack approval gate to the manager -> "
            "approve branch posts the deck link to #leadership; deny branch does "
            "genuinely nothing further (a real dead end is correct here, not a bug) "
            "rather than posting regardless of the approval outcome."
        ),
    ),
    SyntheticInput(
        text=(
            "New Jira tickets in PLAT should get an AI priority assessment. High "
            "priority: alert #oncall in Slack immediately AND transition the ticket to "
            "'In Progress'. Medium: just add a triage comment, no Slack. Low: no Slack, "
            "no comment, just log it in our tracking sheet."
        ),
        category="hard_jira_sla_escalation",
        trigger="jira",
        outputs=["slack_message", "sheets_update"],
        has_approval=False,
        is_ood=False,
        expected_behavior=(
            "Jira Trigger -> AI Agent assesses priority -> 3-way Switch, each branch "
            "doing a DIFFERENT combination of actions rather than all three doing the "
            "same thing: high -> Slack alert AND Jira transition (two actions, both "
            "reachable from that one branch); medium -> Jira comment only; low -> Sheets "
            "log only. All three branches must be reachable and must not share a single "
            "generic 'notify' step that ignores the differences."
        ),
    ),
    SyntheticInput(
        text=(
            "New emails to our shared support inbox should check if we already have an "
            "open tracking row for that email thread in our sheet. If yes, just append "
            "an update note to the existing row. If no, create a brand new row."
        ),
        category="hard_email_thread_dedup",
        trigger="gmail",
        outputs=["sheets_update"],
        has_approval=False,
        is_ood=False,
        expected_behavior=(
            "Gmail Trigger -> a lookup step (e.g. Google Sheets search/filter) checking "
            "whether the thread is already tracked (by threadId or an equivalent stable "
            "key) -> IF found: update/append to the EXISTING row; IF not found: create a "
            "NEW row. Both branches reachable and doing genuinely different sheet "
            "operations — a common failure is always appending a new row regardless of "
            "whether one already exists, silently duplicating tracking rows."
        ),
    ),
]


def load_hard_scenarios() -> list:
    """Returns the fixed hard-scenario list. No generation, no cache — deterministic."""
    return list(HARD_SCENARIOS)
