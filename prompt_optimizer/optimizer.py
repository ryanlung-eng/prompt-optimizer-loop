"""
Generates improved prompt candidates using Claude Haiku,
guided by the worst-performing evaluation examples.
"""
import asyncio
import json
from typing import Dict, List, Tuple

import anthropic

from .config import OptimizerConfig, JudgeConfig
from .judge import EvalResult

_ANALYSIS_SYSTEM = """\
You are a prompt engineer improving an AI workflow builder assistant that helps \
non-technical Ibotta employees automate tasks using n8n.

Supported triggers: Gmail, Slack, Jira, Google Sheets, Cron/Schedule.
Supported outputs: Send Slack message, Send Gmail, Update Google Sheets row.
Optional approval gate: Slack approval queue before any outbound action.

You will receive:
1. The current system prompt
2. Examples where the assistant performed poorly (with judge scores, reasoning, and any hallucinated details)

Identify 2-4 specific, concrete failure patterns. Pay special attention to:
  • knowledge_honesty failures — did the model invent credentials, channel names, or unsupported integrations?
  • OOD failures — did the model try to build unsupported workflows instead of clearly declining?
  • completeness failures — did the model miss explicit details the user provided?
  • approval failures — did the model miss the approval gate requirement?

For each pattern, recommend an EXACT change to the prompt (what to add, remove, or rephrase).
Do not suggest vague improvements.

Return ONLY valid JSON:
{
  "failure_patterns": ["<pattern 1>", "<pattern 2>", ...],
  "recommended_changes": ["<specific change 1>", "<specific change 2>", ...]
}
"""

_IMPROVEMENT_SYSTEM = """\
You are a prompt engineer. Rewrite the given system prompt for an AI workflow builder assistant \
that helps non-technical Slack users build n8n automation workflows.

Apply the requested changes precisely. Keep the improved prompt focused, clear, and complete.
Do NOT add markdown formatting, headers, or bullet points to the prompt itself — write it as \
flowing instructions. Return ONLY the improved prompt text, nothing else.
"""


async def _analyze_failures(
    client: anthropic.AsyncAnthropic,
    model: str,
    current_prompt: str,
    worst: List[EvalResult],
) -> dict:
    examples = "\n\n".join(
        f"Input: {r.input.text}\n"
        f"Category: {r.input.category} | OOD: {r.input.is_ood} | Approval: {r.input.has_approval}\n"
        f"Expected: {r.input.expected_behavior}\n"
        f"Response: {r.actual_response}\n"
        f"Scores: {json.dumps(r.scores)}\n"
        f"Judge reasoning: {json.dumps(r.reasoning)}\n"
        + (f"Hallucinated details: {r.hallucinated_details}" if r.hallucinated_details else "")
        for r in worst
    )
    prompt = (
        f"Current system prompt:\n{current_prompt}\n\n"
        f"Poor-performing examples:\n{examples}"
    )
    msg = await client.messages.create(
        model=model,
        max_tokens=1024,
        system=_ANALYSIS_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(msg.content[0].text)


async def _generate_candidate(
    client: anthropic.AsyncAnthropic,
    model: str,
    current_prompt: str,
    analysis: dict,
    seed_variation: int,
) -> str:
    changes = "\n".join(f"- {c}" for c in analysis.get("recommended_changes", []))
    # Vary the instruction slightly per candidate to get diverse proposals
    angle = ["directly", "step-by-step", "with examples"][seed_variation % 3]
    prompt = (
        f"Current prompt:\n{current_prompt}\n\n"
        f"Apply these changes {angle}:\n{changes}"
    )
    msg = await client.messages.create(
        model=model,
        max_tokens=2048,
        system=_IMPROVEMENT_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


class PromptOptimizer:
    def __init__(self, config: OptimizerConfig, judge_config: JudgeConfig):
        self._config = config
        self._judge_config = judge_config

    def _worst_examples(self, results: List[EvalResult]) -> List[EvalResult]:
        sorted_results = sorted(results, key=lambda r: r.weighted_score)
        return sorted_results[: self._config.worst_examples_k]

    async def generate_candidates(
        self,
        node_name: str,
        current_prompt: str,
        results: List[EvalResult],
    ) -> List[str]:
        """
        Produce N candidate improved prompts for a node, guided by its worst examples.
        Returns list of candidate prompt strings (not including the original).
        """
        worst = self._worst_examples(results)
        client = anthropic.AsyncAnthropic(api_key=None)  # uses ANTHROPIC_API_KEY from env

        print(f"  Analyzing {len(worst)} failure examples for node '{node_name}'…")
        analysis = await _analyze_failures(client, self._config.improvement_model, current_prompt, worst)
        print(f"  Failure patterns: {analysis.get('failure_patterns', [])}")

        tasks = [
            _generate_candidate(client, self._config.improvement_model,
                                current_prompt, analysis, i)
            for i in range(self._config.candidates_per_iteration)
        ]
        candidates = await asyncio.gather(*tasks, return_exceptions=True)

        valid = []
        for c in candidates:
            if isinstance(c, Exception):
                print(f"  Warning: candidate generation failed: {c}")
            else:
                valid.append(c)

        return valid
