"""Config loading with ${ENV_VAR} resolution."""
import os
import re
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional


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
    generation_endpoint: str   # general-purpose LLM for synthetic data + prompt improvement
    judge_endpoint: str        # LLM used for scoring eval results
    eval_endpoint: str         # the KA endpoint being evaluated
    experiment_name: str
    results_table: Optional[str] = None


@dataclass
class SyntheticDataConfig:
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
    candidates_per_iteration: int
    worst_examples_k: int = 5


@dataclass
class Config:
    prompts: Dict[str, str]   # node_name -> current prompt text, pasted in manually
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

    prompts: Dict[str, str] = raw.get("prompts", {})

    db = raw["databricks"]
    databricks = DatabricksConfig(
        workspace_url=db["workspace_url"].rstrip("/"),
        token=db["token"],
        generation_endpoint=db["generation_endpoint"],
        judge_endpoint=db["judge_endpoint"],
        eval_endpoint=db["eval_endpoint"],
        experiment_name=db["experiment_name"],
        results_table=db.get("results_table"),
    )

    sd = raw["synthetic_data"]
    synthetic_data = SyntheticDataConfig(
        num_samples_per_category=sd["num_samples_per_category"],
        categories=sd["categories"],
        cache_path=sd.get("cache_path", ".synthetic_dataset.json"),
    )

    opt = raw["optimizer"]
    optimizer = OptimizerConfig(
        score_threshold=opt["score_threshold"],
        max_iterations=opt["max_iterations"],
        candidates_per_iteration=opt["candidates_per_iteration"],
        worst_examples_k=opt.get("worst_examples_k", 5),
    )

    judge = JudgeConfig(
        dimensions=[JudgeDimension(**d) for d in raw["judge"]["dimensions"]],
    )

    return Config(prompts=prompts, databricks=databricks, synthetic_data=synthetic_data,
                  optimizer=optimizer, judge=judge)
