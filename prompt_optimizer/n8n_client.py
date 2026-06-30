"""
n8n REST API client.
Handles workflow CRUD and prompt extraction/patching.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import N8NConfig, PromptNodeConfig


class N8NClient:
    def __init__(self, config: N8NConfig):
        self._base = config.base_url
        self._headers = {
            "X-N8N-API-KEY": config.api_key,
            "Content-Type": "application/json",
        }
        self._workflow_id = config.workflow_id
        self._prompt_nodes = config.prompt_nodes

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get(self, path: str) -> dict:
        with httpx.Client(headers=self._headers, timeout=30) as client:
            r = client.get(f"{self._base}/api/v1{path}")
            r.raise_for_status()
            return r.json()

    def _put(self, path: str, body: dict) -> dict:
        with httpx.Client(headers=self._headers, timeout=30) as client:
            r = client.put(f"{self._base}/api/v1{path}", json=body)
            r.raise_for_status()
            return r.json()

    @staticmethod
    def _get_nested(obj: Any, path: str) -> Any:
        """Navigate a dot-path with optional [index] notation."""
        parts = re.split(r"\.(?![^\[]*\])", path)
        for part in parts:
            m = re.match(r"^(.+?)\[(\d+)\]$", part)
            if m:
                obj = obj[m.group(1)][int(m.group(2))]
            else:
                obj = obj[part]
        return obj

    @staticmethod
    def _set_nested(obj: Any, path: str, value: Any) -> None:
        """Set a value at a dot-path with optional [index] notation."""
        parts = re.split(r"\.(?![^\[]*\])", path)
        for part in parts[:-1]:
            m = re.match(r"^(.+?)\[(\d+)\]$", part)
            if m:
                obj = obj[m.group(1)][int(m.group(2))]
            else:
                obj = obj[part]

        last = parts[-1]
        m = re.match(r"^(.+?)\[(\d+)\]$", last)
        if m:
            obj[m.group(1)][int(m.group(2))] = value
        else:
            obj[last] = value

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_workflow(self, workflow_id: Optional[str] = None) -> dict:
        wid = workflow_id or self._workflow_id
        return self._get(f"/workflows/{wid}")

    def extract_prompts(self, workflow: dict) -> Dict[str, str]:
        """
        Return {node_name: prompt_text} for all configured prompt nodes.
        Skips nodes not found in the workflow (logs a warning).
        """
        node_map = {n["name"]: n for n in workflow.get("nodes", [])}
        prompts: Dict[str, str] = {}

        for pn in self._prompt_nodes:
            node = node_map.get(pn.node_name)
            if node is None:
                print(f"  Warning: node '{pn.node_name}' not found in workflow")
                continue
            try:
                prompts[pn.node_name] = self._get_nested(node, pn.param_path)
            except (KeyError, IndexError, TypeError) as e:
                print(f"  Warning: could not read '{pn.param_path}' from node '{pn.node_name}': {e}")

        return prompts

    def update_prompts(self, new_prompts: Dict[str, str], dry_run: bool = False) -> None:
        """
        Patch the workflow in-place with updated prompts and PUT it back.
        new_prompts: {node_name: new_prompt_text}
        """
        workflow = self.get_workflow()
        node_map = {n["name"]: n for n in workflow.get("nodes", [])}
        pn_map: Dict[str, PromptNodeConfig] = {pn.node_name: pn for pn in self._prompt_nodes}

        updated = []
        for node_name, new_text in new_prompts.items():
            pn = pn_map.get(node_name)
            node = node_map.get(node_name)
            if pn is None or node is None:
                print(f"  Warning: skipping update for unknown node '{node_name}'")
                continue
            self._set_nested(node, pn.param_path, new_text)
            updated.append(node_name)

        if dry_run:
            print(f"  [dry-run] Would update prompts in nodes: {updated}")
            return

        self._put(f"/workflows/{self._workflow_id}", {
            "name": workflow["name"],
            "nodes": workflow["nodes"],
            "connections": workflow["connections"],
            "active": workflow.get("active", False),
            "settings": workflow.get("settings", {}),
        })
        print(f"  Updated workflow prompts for nodes: {updated}")
