"""
Generates structured synthetic test inputs covering every supported
trigger × output combination, approval sub-workflow variants,
and out-of-distribution requests that should trigger a pushback.

Uses Claude Haiku for generation with a detailed system prompt
so inputs are rich enough for the workflow builder to act without
asking clarifying questions.
"""
import asyncio
import json
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional

import httpx
from tenacity import RetryError, retry, stop_after_attempt, wait_random_exponential

from .config import DatabricksConfig, SyntheticDataConfig


def _unwrap(e: Exception) -> Exception:
    return e.last_attempt.exception() if isinstance(e, RetryError) else e

# ------------------------------------------------------------------ #
# Supported integrations manifest (single source of truth)           #
# ------------------------------------------------------------------ #

SUPPORTED_TRIGGERS = ["cron", "gmail", "slack", "jira", "sheets"]
SUPPORTED_OUTPUTS = ["slack_message", "email", "sheets_update"]
# Outbound outputs that require an optional approval gate
OUTBOUND_OUTPUTS = ["slack_message", "email"]

OOD_SCENARIOS = [
    "Salesforce opportunity stage change",
    "HubSpot contact creation",
    "Twilio/SMS sending",
    "GitHub pull request opened",
    "Stripe payment received",
    "Zoom meeting ended",
    "Figma design updated",
    "Notion database row added",
    "PostgreSQL / MySQL query trigger",
    "Microsoft Teams message",
    "WhatsApp message",
    "Linear issue created",
    "Zendesk ticket opened",
    "DocuSign envelope completed",
]

# All in-distribution combinations to cover. Approval is derived, not stored:
# every outbound send (slack_message, email) always requires it, unconditionally.
# sheets_update writes data rather than sending a message, so it never does.
_COMBINATIONS = [
    # Cron-triggered
    ("cron",   "slack_message"),
    ("cron",   "email"),
    ("cron",   "sheets_update"),
    # Gmail-triggered
    ("gmail",  "slack_message"),
    ("gmail",  "sheets_update"),
    ("gmail",  "email"),
    # Slack-triggered
    ("slack",  "sheets_update"),
    ("slack",  "email"),
    ("slack",  "slack_message"),
    # Jira-triggered
    ("jira",   "slack_message"),
    ("jira",   "email"),
    ("jira",   "sheets_update"),
    # Sheets-triggered
    ("sheets", "slack_message"),
    ("sheets", "email"),
]

# ------------------------------------------------------------------ #
# System prompts                                                      #
# ------------------------------------------------------------------ #

_GEN_SYSTEM = """\
You generate realistic, EXTREMELY DETAILED Slack messages from Ibotta employees \
asking an AI bot to build automation workflows for them. These are non-technical \
business people — they do not know what n8n is.

CRITICAL: Each message must include ALL of the following so the workflow builder \
can act WITHOUT asking any clarifying questions:
  • The exact Gmail account(s) involved (e.g. ryan.lung@ibotta.com)
  • The exact Slack channel(s) and workspace (Ibotta workspace)
  • The exact trigger condition — for schedules, include day, time, and timezone (MT)
  • The exact data fields to read/write and how they map between systems
  • Business context: frequency, time currently wasted, who does this manually now
  • If multiple steps, the order and any conditions between them

Do NOT have the user offer, ask for, or mention their own Slack ID/handle, or \
name a specific approver — the workflow builder already knows who the requester \
is from context, and approval always routes automatically to the workflow owner. \
The user should never need to say who approves it.

Supported integrations (ONLY these exist — do not invent others):
  TRIGGERS : Gmail (new email matching conditions), Slack message, Google Sheets \
(new/updated row), Jira issue event, Cron/Schedule
  OUTPUTS  : Send Slack message to a channel, Send Gmail, Update Google Sheets row
  APPROVAL : Every Slack message or email send is MANDATORY-approved — the \
requester automatically gets a Slack DM with Approve/Deny buttons and a preview \
before it sends. This is not optional and the user never asks for it themselves. \
Sheets updates are not messages sent to a person, so they do not go through approval.

Tone: conversational Slack DM, may have mild typos, 3–10 sentences.
Return ONLY a valid JSON array of strings, no other text."""

_OOD_SYSTEM = """\
You generate realistic Slack messages from Ibotta employees requesting automations \
that CANNOT be built with the available tools.

Unsupported integrations (use these as the core request): \
{unsupported_list}

Rules:
  • Sound like genuine business requests from a real employee
  • Be specific — name real-sounding systems and business processes
  • Do NOT combine with supported integrations as a workaround in the message
  • The workflow described must fundamentally require the unsupported system

Tone: same conversational Slack DM style as any other request.
Return ONLY a valid JSON array of strings, no other text."""

_BEHAVIOR_SYSTEM = """\
You describe, in ONE sentence, what a high-quality workflow builder assistant \
should do in response to each user message.

For supported workflows: describe the exact workflow structure, key nodes, \
and any approval gates.
For UNSUPPORTED workflows: say the assistant should clearly explain the \
integration is not available, list what IS supported, and suggest the closest \
supported alternative if one exists.

Return ONLY a valid JSON array of strings, one per input message."""


# ------------------------------------------------------------------ #
# Data model                                                         #
# ------------------------------------------------------------------ #

@dataclass
class SyntheticInput:
    text: str
    category: str              # e.g. "cron_to_slack_message"
    trigger: Optional[str]     # None for OOD
    outputs: List[str]
    has_approval: bool
    is_ood: bool
    expected_behavior: str
    time_saved_minutes: int = 10   # matches the real "AI Agent" node's pre-computed estimate

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SyntheticInput":
        return cls(**d)


# ------------------------------------------------------------------ #
# Generation helpers                                                  #
# ------------------------------------------------------------------ #

def _combo_label(trigger: str, output: str) -> str:
    return f"{trigger}_to_{output}"


def _combo_user_prompt(trigger: str, output: str, n: int) -> str:
    approval_note = (
        " The requester will automatically get a Slack approval DM (Approve/Deny "
        "with a preview) before this send fires — mention that as expected behavior, "
        "not something the user needs to ask for."
        if output in OUTBOUND_OUTPUTS else ""
    )
    return (
        f"Generate {n} messages. The automation is: triggered by {trigger.upper()}, "
        f"outputs to {output.upper().replace('_', ' ')}.{approval_note}\n\n"
        "Remember: include EVERY detail so the builder needs zero clarification. "
        f"Vary the business scenario across the {n} messages."
    )


@retry(stop=stop_after_attempt(6), wait=wait_random_exponential(multiplier=1, min=4, max=60))
async def _db_call(
    client: httpx.AsyncClient,
    endpoint_url: str,
    headers: dict,
    system: str,
    user: str,
    max_tokens: int = 2048,
) -> str:
    resp = await client.post(
        endpoint_url,
        headers=headers,
        json={
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=90,
    )
    if resp.status_code >= 400:
        raise ValueError(f"{resp.status_code} from {endpoint_url}: {resp.text[:1500]}")
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    if not content or not content.strip():
        raise ValueError(
            f"Empty content from {endpoint_url}. "
            f"finish_reason={body['choices'][0].get('finish_reason')!r}. "
            f"Raw response: {json.dumps(body)[:1500]}"
        )
    # Extract the outermost [...] array — handles stray prose or fences
    # anywhere around the JSON, not just at the exact string boundaries.
    # Every caller of _db_call in this module expects a JSON array.
    start, end = content.find("["), content.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON array found in response: {content[:1500]}")
    return content[start:end + 1]


async def _generate_combo(
    endpoint_url: str,
    headers: dict,
    trigger: str,
    output: str,
    config: SyntheticDataConfig,
) -> List[SyntheticInput]:
    n = config.num_samples_per_category
    approval = output in OUTBOUND_OUTPUTS
    user_prompt = _combo_user_prompt(trigger, output, n)

    async with httpx.AsyncClient() as client:
        raw_texts = await _db_call(client, endpoint_url, headers, _GEN_SYSTEM, user_prompt)
        texts: List[str] = json.loads(raw_texts)

        raw_behaviors = await _db_call(
            client, endpoint_url, headers, _BEHAVIOR_SYSTEM,
            f"User messages:\n{json.dumps(texts, indent=2)}", max_tokens=2048,
        )
        behaviors: List[str] = json.loads(raw_behaviors)

    category = _combo_label(trigger, output)
    return [
        SyntheticInput(
            text=t,
            category=category,
            trigger=trigger,
            outputs=[output],
            has_approval=approval,
            is_ood=False,
            expected_behavior=b,
            time_saved_minutes=random.randint(5, 60),
        )
        for t, b in zip(texts, behaviors)
    ]


async def _generate_ood(
    endpoint_url: str,
    headers: dict,
    config: SyntheticDataConfig,
    n: int = 3,
) -> List[SyntheticInput]:
    """Generate OOD inputs — one batch per unsupported scenario."""
    results: List[SyntheticInput] = []

    for scenario in OOD_SCENARIOS:
        system = _OOD_SYSTEM.format(unsupported_list=scenario)
        try:
            async with httpx.AsyncClient() as client:
                raw_texts = await _db_call(
                    client, endpoint_url, headers, system,
                    f"Generate {n} messages requesting automation around: {scenario}",
                    max_tokens=1500,
                )
                texts: List[str] = json.loads(raw_texts)

                raw_behaviors = await _db_call(
                    client, endpoint_url, headers, _BEHAVIOR_SYSTEM,
                    f"User messages:\n{json.dumps(texts, indent=2)}",
                    max_tokens=1024,
                )
                behaviors: List[str] = json.loads(raw_behaviors)

            results.extend([
                SyntheticInput(
                    text=t,
                    category=f"ood_{scenario.split('/')[0].split(' ')[0].lower()}",
                    trigger=None,
                    outputs=[],
                    has_approval=False,
                    is_ood=True,
                    expected_behavior=b,
                )
                for t, b in zip(texts, behaviors)
            ])
        except Exception as e:
            print(f"  Warning: OOD generation failed for '{scenario}': {_unwrap(e)}")

    return results


# ------------------------------------------------------------------ #
# Public entry point                                                  #
# ------------------------------------------------------------------ #

async def generate_dataset(config: SyntheticDataConfig, db_config: DatabricksConfig) -> List[SyntheticInput]:
    """
    Generate the full dataset. Cached to disk — delete the cache file to regenerate.
    Returns inputs sorted so in-distribution come first, OOD at the end.
    """
    cache = Path(config.cache_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        print(f"  Loading synthetic dataset from cache: {cache}")
        raw = json.loads(cache.read_text())
        inputs = [SyntheticInput.from_dict(d) for d in raw]
        ood_count = sum(1 for i in inputs if i.is_ood)
        print(f"  {len(inputs)} total inputs ({len(inputs) - ood_count} in-dist, {ood_count} OOD)")
        return inputs

    print(f"  Generating synthetic dataset: {len(_COMBINATIONS)} trigger×output combinations…")
    endpoint_url = f"{db_config.workspace_url}/serving-endpoints/{db_config.generation_endpoint}/invocations"
    headers = {"Authorization": f"Bearer {db_config.token}", "Content-Type": "application/json"}

    combo_tasks = [
        _generate_combo(endpoint_url, headers, trigger, output, config)
        for trigger, output in _COMBINATIONS
    ]
    combo_results = await asyncio.gather(*combo_tasks, return_exceptions=True)

    inputs: List[SyntheticInput] = []
    for (trigger, output), result in zip(_COMBINATIONS, combo_results):
        label = _combo_label(trigger, output)
        if isinstance(result, Exception):
            print(f"  Warning: failed to generate combo '{label}': {_unwrap(result)}")
        else:
            inputs.extend(result)

    # OOD refusal is the earlier conversation node's job, not Workflow Builder's —
    # it should never even reach this node. _generate_ood/OOD_SCENARIOS are kept
    # in this module, unused, for whenever that node gets its own optimizer.
    cache.write_text(json.dumps([i.to_dict() for i in inputs], indent=2))
    print(f"  Total: {len(inputs)} inputs → cached to {cache}")
    return inputs
