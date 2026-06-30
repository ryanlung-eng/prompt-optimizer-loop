"""
LLM-as-judge using Databricks model serving.

Scores workflow builder responses on five dimensions:
  intent_understanding  — understood what workflow the user wants
  workflow_accuracy     — correct trigger/action/node structure (or correct refusal for OOD)
  clarity               — understandable to a non-technical user
  completeness          — all user-specified details addressed
  knowledge_honesty     — no hallucinated credentials/integrations; admits gaps clearly

OOD inputs get a tailored judge prompt that evaluates pushback quality instead
of workflow construction. knowledge_honesty is the most important dimension for OOD.
"""
import asyncio
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import DatabricksConfig, JudgeConfig, JudgeDimension
from .synthetic_data import SyntheticInput

# ------------------------------------------------------------------ #
# Judge prompts                                                       #
# ------------------------------------------------------------------ #

_JUDGE_SYSTEM_IN_DIST = """\
You are an expert evaluator of an AI workflow builder assistant. \
The assistant helps non-technical Ibotta employees build automation workflows \
(using Gmail, Slack, Jira, Google Sheets, and Cron triggers; \
with outputs of Slack messages, emails, and Sheets updates; \
with optional Slack approval gates for outbound actions).

You will receive: the user's original request, what a great response should do, \
and the actual assistant response.

Score each dimension 0.0–1.0 using this rubric:

intent_understanding  — 1.0: Correctly identified the trigger, output(s), and approval requirements.
                        0.5: Got the gist but missed one component. 0.0: Misunderstood.

workflow_accuracy     — 1.0: The workflow structure is technically correct for the n8n nodes available.
                        Includes correct trigger config, action config, and approval sub-workflow if needed.
                        0.5: Mostly correct but a node misconfigured. 0.0: Wrong structure.

clarity               — 1.0: Clear, step-by-step, plain language a non-tech user can follow.
                        0.5: Understandable but with jargon or missing steps. 0.0: Confusing.

completeness          — 1.0: Every detail the user provided (account, channel, schedule, data mapping)
                        is reflected in the proposed workflow. Nothing left vague.
                        0.5: Some details addressed. 0.0: Ignored user specifics.

knowledge_honesty     — 1.0: No invented credentials, account names, or node configs. When any
                        detail was unclear, the assistant asked rather than assumed. No hallucination.
                        0.5: Mostly honest, but assumed 1–2 details not in the user message.
                        0.0: Fabricated credentials, channels, or workflow steps not in the request.

Return ONLY valid JSON — no markdown, no prose:
{
  "scores": {"intent_understanding": float, "workflow_accuracy": float,
             "clarity": float, "completeness": float, "knowledge_honesty": float},
  "reasoning": {"intent_understanding": "str", "workflow_accuracy": "str",
                "clarity": "str", "completeness": "str", "knowledge_honesty": "str"},
  "hallucinated_details": ["<specific thing the assistant made up>", ...],
  "overall_comment": "str"
}"""

_JUDGE_SYSTEM_OOD = """\
You are evaluating whether an AI workflow builder correctly handles a request \
for an UNSUPPORTED integration.

Supported integrations: Gmail trigger, Slack trigger, Jira trigger, Google Sheets trigger, \
Cron/schedule trigger; outputs: Slack message, Gmail, Sheets row update; \
Slack approval gate.

The user's request requires a system NOT in that list. A correct response should:
  1. Clearly state the specific integration is not supported (no hedging)
  2. List what IS available without being condescending
  3. Suggest a closest supported alternative when one exists
  4. NOT attempt to partially build the unsupported workflow

Score dimensions 0.0–1.0:

intent_understanding  — 1.0: Correctly identified this as an unsupported request.
                        0.0: Tried to build it anyway or misread the request.

workflow_accuracy     — 1.0: Accurate description of why it can't be done + what is supported.
                        0.0: Hallucinated a workaround using non-existent integrations.

clarity               — 1.0: Pushback is clear, polite, and actionable.

completeness          — 1.0: Listed supported alternatives and offered a workaround if applicable.
                        0.0: Just said "can't do that" with no next steps.

knowledge_honesty     — 1.0: Honest about capability limits, no false promises, no invented integrations.
                        0.0: Claimed it could partially support the unsupported system.

Return ONLY valid JSON (same schema as in-distribution judge)."""

_JUDGE_USER_TEMPLATE = """\
USER REQUEST:
{user_message}

EXPECTED BEHAVIOR:
{expected_behavior}

ACTUAL ASSISTANT RESPONSE:
{actual_response}"""


# ------------------------------------------------------------------ #
# Data model                                                         #
# ------------------------------------------------------------------ #

@dataclass
class EvalResult:
    input: SyntheticInput
    actual_response: str
    scores: Dict[str, float]
    reasoning: Dict[str, str]
    hallucinated_details: List[str]
    overall_comment: str
    weighted_score: float = field(default=0.0, init=False)

    def __post_init__(self):
        pass   # weighted_score set by judge after creation


def _weighted_score(scores: Dict[str, float], dimensions: List[JudgeDimension]) -> float:
    total_weight = sum(d.weight for d in dimensions)
    return sum(
        scores.get(d.name, 0.0) * d.weight for d in dimensions
    ) / max(total_weight, 1e-9)


# ------------------------------------------------------------------ #
# Judge class                                                        #
# ------------------------------------------------------------------ #

class DatabricksJudge:
    def __init__(self, config: DatabricksConfig, judge_config: JudgeConfig):
        self._config = config
        self._judge_config = judge_config
        self._endpoint_url = (
            f"{config.workspace_url}/serving-endpoints/{config.judge_endpoint}/invocations"
        )
        self._headers = {
            "Authorization": f"Bearer {config.token}",
            "Content-Type": "application/json",
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    async def _call(
        self,
        client: httpx.AsyncClient,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        resp = await client.post(
            self._endpoint_url,
            headers=self._headers,
            json={
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 1200,
                "temperature": 0.0,
            },
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        # Strip markdown fences if the model wraps JSON anyway
        content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(content)

    async def evaluate_one(
        self,
        client: httpx.AsyncClient,
        inp: SyntheticInput,
        actual_response: str,
    ) -> EvalResult:
        system = _JUDGE_SYSTEM_OOD if inp.is_ood else _JUDGE_SYSTEM_IN_DIST
        user = _JUDGE_USER_TEMPLATE.format(
            user_message=inp.text,
            expected_behavior=inp.expected_behavior,
            actual_response=actual_response,
        )
        try:
            parsed = await self._call(client, system, user)
            scores = parsed.get("scores", {})
            reasoning = parsed.get("reasoning", {})
            hallucinated = parsed.get("hallucinated_details", [])
            comment = parsed.get("overall_comment", "")
        except Exception as e:
            print(f"  Warning: judge failed for '{inp.text[:60]}…': {e}")
            scores = {d.name: 0.0 for d in self._judge_config.dimensions}
            reasoning = {d.name: f"Judge error: {e}" for d in self._judge_config.dimensions}
            hallucinated = []
            comment = f"Judge error: {e}"

        result = EvalResult(
            input=inp,
            actual_response=actual_response,
            scores=scores,
            reasoning=reasoning,
            hallucinated_details=hallucinated,
            overall_comment=comment,
        )
        result.weighted_score = _weighted_score(scores, self._judge_config.dimensions)
        return result

    async def evaluate_batch(
        self,
        inputs_and_responses: List[Tuple[SyntheticInput, str]],
    ) -> List[EvalResult]:
        """Evaluate a batch concurrently, respecting a semaphore to avoid rate limits."""
        sem = asyncio.Semaphore(8)

        async def bounded(inp: SyntheticInput, resp: str) -> EvalResult:
            async with sem:
                async with httpx.AsyncClient() as client:
                    return await self.evaluate_one(client, inp, resp)

        return await asyncio.gather(*[bounded(inp, resp) for inp, resp in inputs_and_responses])
