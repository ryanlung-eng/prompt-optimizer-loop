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
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional

import anthropic

from .config import SyntheticDataConfig

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

# All in-distribution combinations to cover
# (trigger, output, with_approval)
_COMBINATIONS = [
    # Cron-triggered
    ("cron",   "slack_message",  False),
    ("cron",   "email",          False),
    ("cron",   "sheets_update",  False),
    # Gmail-triggered
    ("gmail",  "slack_message",  False),
    ("gmail",  "sheets_update",  False),
    ("gmail",  "email",          True),   # forward/reply needs approval
    # Slack-triggered
    ("slack",  "sheets_update",  False),
    ("slack",  "email",          True),
    ("slack",  "slack_message",  False),  # cross-channel post
    # Jira-triggered
    ("jira",   "slack_message",  False),
    ("jira",   "email",          True),
    ("jira",   "sheets_update",  False),
    # Sheets-triggered
    ("sheets", "slack_message",  False),
    ("sheets", "email",          True),
    ("sheets", "slack_message",  True),   # approval before posting to Slack
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
  • Whether someone needs to approve an action before it's sent

Supported integrations (ONLY these exist — do not invent others):
  TRIGGERS : Gmail (new email matching conditions), Slack message, Google Sheets \
(new/updated row), Jira issue event, Cron/Schedule
  OUTPUTS  : Send Slack message to a channel, Send Gmail, Update Google Sheets row
  APPROVAL : A Slack approval queue can gate any outbound action — specify who \
the approver(s) are

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

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SyntheticInput":
        return cls(**d)


# ------------------------------------------------------------------ #
# Generation helpers                                                  #
# ------------------------------------------------------------------ #

def _combo_label(trigger: str, output: str, approval: bool) -> str:
    label = f"{trigger}_to_{output}"
    return f"{label}_with_approval" if approval else label


def _combo_user_prompt(trigger: str, output: str, approval: bool, n: int) -> str:
    approval_note = (
        " Include an explicit Slack approval gate (with named approver) "
        "before the outbound action is taken." if approval else ""
    )
    return (
        f"Generate {n} messages. The automation is: triggered by {trigger.upper()}, "
        f"outputs to {output.upper().replace('_', ' ')}.{approval_note}\n\n"
        "Remember: include EVERY detail so the builder needs zero clarification. "
        "Vary the business scenario across the {n} messages."
    )


async def _generate_combo(
    client: anthropic.AsyncAnthropic,
    trigger: str,
    output: str,
    approval: bool,
    config: SyntheticDataConfig,
) -> List[SyntheticInput]:
    n = config.num_samples_per_category
    user_prompt = _combo_user_prompt(trigger, output, approval, n)

    msg_resp = await client.messages.create(
        model=config.model,
        max_tokens=2048,
        system=_GEN_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )
    texts: List[str] = json.loads(msg_resp.content[0].text)

    beh_resp = await client.messages.create(
        model=config.model,
        max_tokens=1024,
        system=_BEHAVIOR_SYSTEM,
        messages=[{"role": "user", "content": f"User messages:\n{json.dumps(texts, indent=2)}"}],
    )
    behaviors: List[str] = json.loads(beh_resp.content[0].text)

    category = _combo_label(trigger, output, approval)
    return [
        SyntheticInput(
            text=t,
            category=category,
            trigger=trigger,
            outputs=[output],
            has_approval=approval,
            is_ood=False,
            expected_behavior=b,
        )
        for t, b in zip(texts, behaviors)
    ]


async def _generate_ood(
    client: anthropic.AsyncAnthropic,
    config: SyntheticDataConfig,
    n: int = 3,
) -> List[SyntheticInput]:
    """Generate OOD inputs — one batch per unsupported scenario."""
    results: List[SyntheticInput] = []

    for scenario in OOD_SCENARIOS:
        system = _OOD_SYSTEM.format(unsupported_list=scenario)
        try:
            msg_resp = await client.messages.create(
                model=config.model,
                max_tokens=800,
                system=system,
                messages=[{
                    "role": "user",
                    "content": f"Generate {n} messages requesting automation around: {scenario}",
                }],
            )
            texts: List[str] = json.loads(msg_resp.content[0].text)

            beh_resp = await client.messages.create(
                model=config.model,
                max_tokens=512,
                system=_BEHAVIOR_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": f"User messages:\n{json.dumps(texts, indent=2)}",
                }],
            )
            behaviors: List[str] = json.loads(beh_resp.content[0].text)

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
            print(f"  Warning: OOD generation failed for '{scenario}': {e}")

    return results


# ------------------------------------------------------------------ #
# Public entry point                                                  #
# ------------------------------------------------------------------ #

async def generate_dataset(config: SyntheticDataConfig) -> List[SyntheticInput]:
    """
    Generate the full dataset. Cached to disk — delete the cache file to regenerate.
    Returns inputs sorted so in-distribution come first, OOD at the end.
    """
    cache = Path(config.cache_path)
    if cache.exists():
        print(f"  Loading synthetic dataset from cache: {cache}")
        raw = json.loads(cache.read_text())
        inputs = [SyntheticInput.from_dict(d) for d in raw]
        ood_count = sum(1 for i in inputs if i.is_ood)
        print(f"  {len(inputs)} total inputs ({len(inputs) - ood_count} in-dist, {ood_count} OOD)")
        return inputs

    print(
        f"  Generating synthetic dataset: {len(_COMBINATIONS)} trigger×output combinations "
        f"+ {len(OOD_SCENARIOS)} OOD scenarios…"
    )
    client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

    combo_tasks = [
        _generate_combo(client, trigger, output, approval, config)
        for trigger, output, approval in _COMBINATIONS
    ]
    combo_results = await asyncio.gather(*combo_tasks, return_exceptions=True)

    inputs: List[SyntheticInput] = []
    for (trigger, output, approval), result in zip(_COMBINATIONS, combo_results):
        label = _combo_label(trigger, output, approval)
        if isinstance(result, Exception):
            print(f"  Warning: failed to generate combo '{label}': {result}")
        else:
            inputs.extend(result)

    print(f"  Generated {len(inputs)} in-distribution inputs. Now generating OOD…")
    ood = await _generate_ood(client, config)
    inputs.extend(ood)

    cache.write_text(json.dumps([i.to_dict() for i in inputs], indent=2))
    print(
        f"  Total: {len(inputs)} inputs "
        f"({len(inputs) - len(ood)} in-dist, {len(ood)} OOD) → cached to {cache}"
    )
    return inputs
