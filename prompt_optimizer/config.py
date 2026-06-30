"""Config loading with ${ENV_VAR} resolution."""
import os
import re
import yaml
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PromptNodeConfig:
    node_name: str
    param_path: str
    description: str = ""


@dataclass
class N8NConfig:
    base_url: str
    api_key: str
    workflow_id: str
    prompt_nodes: List[PromptNodeConfig] = field(default_factory=list)


@dataclass
class DatabricksConfig:
    workspace_url: str
    token: str
    judge_endpoint: str
    eval_endpoint: str
    experiment_name: str
    results_table: Optional[str] = None


@dataclass
class SyntheticDataConfig:
    anthropic_api_key: str
    model: str
    num_samples_per_category: int
    categories: List[str]
    cache_path: str = ".synthetic_dataset.json"


@dataclass
class JudgeDimension:
    name: str
    weight: float
    description: str


@dataclass
class JudgeConfig:
    dimensions: List[JudgeDimension]


@dataclass
class OptimizerConfig:
    score_threshold: float
    max_iterations: int
    improvement_model: str
    candidates_per_iteration: int
    worst_examples_k: int = 5
    dry_run: bool = False


@dataclass
class Config:
    n8n: N8NConfig
    databricks: DatabricksConfig
    synthetic_data: SyntheticDataConfig
    optimizer: OptimizerConfig
    judge: JudgeConfig


def _resolve(value: str) -> str:
    """Replace ${VAR} tokens with environment variable values."""
    def replace(match):
        var = match.group(1)
        val = os.environ.get(var)
        if val is None:
            raise EnvironmentError(f"Required env var not set: {var}")
        return val

    return re.sub(r"\$\{([^}]+)\}", replace, value) if isinstance(value, str) else value


def _walk(obj):
    if isinstance(obj, dict):
        return {k: _walk(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk(i) for i in obj]
    if isinstance(obj, str):
        return _resolve(obj)
    return obj


def load_config(path: str = "config.yaml") -> Config:
    with open(path) as f:
        raw = _walk(yaml.safe_load(f))

    n8n = N8NConfig(
        base_url=raw["n8n"]["base_url"].rstrip("/"),
        api_key=raw["n8n"]["api_key"],
        workflow_id=raw["n8n"]["workflow_id"],
        prompt_nodes=[PromptNodeConfig(**pn) for pn in raw["n8n"].get("prompt_nodes", [])],
    )

    db = raw["databricks"]
    databricks = DatabricksConfig(
        workspace_url=db["workspace_url"].rstrip("/"),
        token=db["token"],
        judge_endpoint=db["judge_endpoint"],
        eval_endpoint=db["eval_endpoint"],
        experiment_name=db["experiment_name"],
        results_table=db.get("results_table"),
    )

    sd = raw["synthetic_data"]
    synthetic_data = SyntheticDataConfig(
        anthropic_api_key=sd["anthropic_api_key"],
        model=sd["model"],
        num_samples_per_category=sd["num_samples_per_category"],
        categories=sd["categories"],
        cache_path=sd.get("cache_path", ".synthetic_dataset.json"),
    )

    opt = raw["optimizer"]
    optimizer = OptimizerConfig(
        score_threshold=opt["score_threshold"],
        max_iterations=opt["max_iterations"],
        improvement_model=opt["improvement_model"],
        candidates_per_iteration=opt["candidates_per_iteration"],
        worst_examples_k=opt.get("worst_examples_k", 5),
        dry_run=opt.get("dry_run", False),
    )

    judge = JudgeConfig(
        dimensions=[JudgeDimension(**d) for d in raw["judge"]["dimensions"]],
    )

    return Config(n8n=n8n, databricks=databricks, synthetic_data=synthetic_data,
                  optimizer=optimizer, judge=judge)
