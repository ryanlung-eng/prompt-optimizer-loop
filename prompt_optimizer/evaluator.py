"""
Evaluates a set of prompts by calling Databricks directly
(bypasses Slack/n8n trigger complexity for tight eval loops).
"""
import asyncio
import json
from typing import Dict, List, Tuple

import httpx
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from .config import DatabricksConfig
from .synthetic_data import SyntheticInput

_MAX_CONCURRENT = 10  # Cap concurrent Databricks calls

# In real n8n, these three expressions are resolved by n8n's own expression
# engine before the LLM ever sees the prompt — the model never sees literal
# "{{ }}" syntax in production. We resolve the same three here so the eval
# harness matches what the KA actually receives at runtime.
_SYNTHETIC_SLACK_USER_ID = "U0EVAL0001"
_CONVERSATION_EXPR = "{{ $('Thread Formatter').item.json.conversation }}"
_USER_ID_EXPR = "{{ $('Slack Trigger').item.json.user }}"
_TIME_SAVED_EXPR = "{{ $('AI Agent').item.json.output.time_saved }}"


def _resolve_prompt(system_prompt: str, inp: SyntheticInput) -> str:
    return (
        system_prompt
        .replace(_CONVERSATION_EXPR, inp.text)
        .replace(_USER_ID_EXPR, _SYNTHETIC_SLACK_USER_ID)
        .replace(_TIME_SAVED_EXPR, str(inp.time_saved_minutes))
    )


class WorkflowEvaluator:
    """
    Calls the Databricks model serving endpoint directly with a given system
    prompt + synthetic user message, returning the model's raw response.
    This lets us evaluate prompt changes without triggering the full Slack workflow.
    """

    def __init__(self, config: DatabricksConfig):
        self._config = config
        self._endpoint_url = (
            f"{config.workspace_url}/serving-endpoints/{config.eval_endpoint}/invocations"
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
        user_message: str,
    ) -> str:
        resp = await client.post(
            self._endpoint_url,
            headers=self._headers,
            json={
                "input": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": 1500,
                "temperature": 0.3,
            },
            timeout=90,
        )
        if resp.status_code >= 400:
            raise ValueError(
                f"{resp.status_code} from {self._endpoint_url}: {resp.text[:1500]}"
            )
        body = resp.json()

        if "choices" in body:
            return body["choices"][0]["message"]["content"]
        if "output" in body:
            for item in body["output"]:
                if item.get("type") == "message":
                    for block in item.get("content", []):
                        if block.get("type") in ("output_text", "text") and block.get("text"):
                            return block["text"]
        raise ValueError(
            f"Unrecognized response shape from {self._endpoint_url}: {json.dumps(body)[:1500]}"
        )

    async def run_batch(
        self,
        system_prompt: str,
        inputs: List[SyntheticInput],
    ) -> List[Tuple[SyntheticInput, str]]:
        """
        Run all inputs against a single system prompt concurrently.
        Returns [(input, model_response), …]
        """
        sem = asyncio.Semaphore(_MAX_CONCURRENT)

        async def bounded_call(inp: SyntheticInput) -> Tuple[SyntheticInput, str]:
            async with sem:
                async with httpx.AsyncClient() as client:
                    try:
                        resolved = _resolve_prompt(system_prompt, inp)
                        response = await self._call(client, resolved, inp.text)
                    except Exception as e:
                        cause = e.last_attempt.exception() if isinstance(e, RetryError) else e
                        print(f"  Warning: eval failed for '{inp.text[:60]}…': {cause}")
                        response = f"[ERROR: {cause}]"
                    return inp, response

        return await asyncio.gather(*[bounded_call(inp) for inp in inputs])

    async def run_multi_prompt_batch(
        self,
        node_prompts: Dict[str, str],
        inputs: List[SyntheticInput],
    ) -> Dict[str, List[Tuple[SyntheticInput, str]]]:
        """
        Evaluate multiple nodes' prompts simultaneously.
        Returns {node_name: [(input, response), …]}
        """
        tasks = {
            node_name: self.run_batch(prompt, inputs)
            for node_name, prompt in node_prompts.items()
        }
        results = await asyncio.gather(*tasks.values())
        return dict(zip(tasks.keys(), results))
