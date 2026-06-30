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
                "messages": [
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
        return resp.json()["choices"][0]["message"]["content"]

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
                        response = await self._call(client, system_prompt, inp.text)
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
