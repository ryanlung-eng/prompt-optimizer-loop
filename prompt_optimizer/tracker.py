"""
MLflow experiment tracking for prompt versions and scores.
Also generates a gap report: hallucination inventory and OOD pushback analysis.
"""
import json
import os
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

from .config import DatabricksConfig
from .judge import EvalResult


@dataclass
class IterationSummary:
    iteration: int
    node_name: str
    prompt_version: str
    prompt_text: str
    overall_score: float
    dim_scores: Dict[str, float]
    num_inputs: int
    run_id: str


@dataclass
class GapReport:
    """Surfaces knowledge base holes identified during evaluation."""
    hallucinated_details: List[str]          # things the model made up
    ood_pushback_failures: List[str]         # OOD inputs the model tried to build anyway
    low_honesty_categories: List[str]        # combo categories where honesty score < 0.6
    ood_correctly_refused: int
    ood_attempted_build: int
    avg_honesty_score: float


class PromptTracker:
    def __init__(self, config: DatabricksConfig):
        os.environ["DATABRICKS_HOST"] = config.workspace_url
        os.environ["DATABRICKS_TOKEN"] = config.token
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment(config.experiment_name)
        self._client = MlflowClient()
        self._experiment_name = config.experiment_name

    @contextmanager
    def start_iteration(
        self,
        iteration: int,
        node_name: str,
        prompt_text: str,
        prompt_version: str,
        tags: Optional[Dict[str, str]] = None,
    ):
        run_tags = {
            "iteration": str(iteration),
            "node_name": node_name,
            "prompt_version": prompt_version,
            **(tags or {}),
        }
        with mlflow.start_run(
            run_name=f"iter{iteration}_{node_name}_{prompt_version}",
            tags=run_tags,
        ) as run:
            mlflow.log_param("node_name", node_name)
            mlflow.log_param("prompt_version", prompt_version)
            mlflow.log_param("iteration", iteration)
            mlflow.log_text(prompt_text, "prompt.txt")
            yield run

    def log_results(
        self,
        run: mlflow.ActiveRun,
        results: List[EvalResult],
        dim_names: List[str],
    ) -> Optional[IterationSummary]:
        if not results:
            return None

        overall_scores = [r.weighted_score for r in results]
        avg_overall = sum(overall_scores) / len(overall_scores)

        dim_avgs: Dict[str, float] = {
            d: sum(r.scores.get(d, 0.0) for r in results) / max(len(results), 1)
            for d in dim_names
        }

        mlflow.log_metric("avg_overall_score", avg_overall)
        mlflow.log_metric("num_inputs", len(results))
        for dim, val in dim_avgs.items():
            mlflow.log_metric(f"avg_{dim}", val)

        sorted_scores = sorted(overall_scores)
        mlflow.log_metric("score_p25", sorted_scores[len(sorted_scores) // 4])
        mlflow.log_metric("score_p75", sorted_scores[3 * len(sorted_scores) // 4])

        # Did the KA ever attempt JSON, and did it end up structurally valid?
        # Deterministic — no LLM judge involved, answers "does the output actually work"
        # directly rather than via an LLM's subjective read of the response text.
        # Mutually exclusive with pct_structurally_valid — see EvalResult.ever_attempted_json
        # for why the self-repair loop makes "attempted" and "valid" no longer redundant.
        turn_counts = [len([t for t in r.transcript if t["role"] == "ka"]) for r in results if r.transcript]
        mlflow.log_metric(
            "pct_structurally_valid",
            sum(1 for r in results if r.structural.valid) / max(len(results), 1),
        )
        mlflow.log_metric(
            "pct_attempted_but_invalid",
            sum(1 for r in results if r.ever_attempted_json and not r.structural.valid) / max(len(results), 1),
        )
        mlflow.log_metric(
            "avg_structural_score",
            sum(r.structural.score for r in results) / max(len(results), 1),
        )
        if turn_counts:
            mlflow.log_metric("avg_turns_to_resolution", sum(turn_counts) / len(turn_counts))

        # OOD-specific metrics
        ood = [r for r in results if r.input.is_ood]
        if ood:
            ood_refused = sum(
                1 for r in ood if r.scores.get("intent_understanding", 0) >= 0.7
            )
            mlflow.log_metric("ood_correct_refusals", ood_refused)
            mlflow.log_metric("ood_total", len(ood))

        # Full eval detail artifact
        details = [
            {
                "category": r.input.category,
                "is_ood": r.input.is_ood,
                "trigger": r.input.trigger,
                "outputs": r.input.outputs,
                "has_approval": r.input.has_approval,
                "input": r.input.text,
                "expected": r.input.expected_behavior,
                "response": r.actual_response,
                "scores": r.scores,
                "reasoning": r.reasoning,
                "hallucinated_details": r.hallucinated_details,
                "weighted_score": r.weighted_score,
                "overall_comment": r.overall_comment,
                "transcript": r.transcript,
                "turns": len([t for t in r.transcript if t["role"] == "ka"]),
                "structural_valid": r.structural.valid,
                "structural_checks": r.structural.checks,
                "structural_errors": r.structural.errors,
            }
            for r in results
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(details, f, indent=2)
            tmp = f.name
        mlflow.log_artifact(tmp, artifact_path="eval_results")
        os.unlink(tmp)

        # Gap report artifact
        gap = self._build_gap_report(results)
        gap_dict = {
            "hallucinated_details": gap.hallucinated_details,
            "ood_pushback_failures": gap.ood_pushback_failures,
            "low_honesty_categories": gap.low_honesty_categories,
            "ood_correctly_refused": gap.ood_correctly_refused,
            "ood_attempted_build": gap.ood_attempted_build,
            "avg_honesty_score": gap.avg_honesty_score,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(gap_dict, f, indent=2)
            tmp = f.name
        mlflow.log_artifact(tmp, artifact_path="gap_report")
        os.unlink(tmp)

        return IterationSummary(
            iteration=int(run.info.tags.get("iteration", 0)),
            node_name=run.info.tags.get("node_name", ""),
            prompt_version=run.info.tags.get("prompt_version", ""),
            prompt_text="",
            overall_score=avg_overall,
            dim_scores=dim_avgs,
            num_inputs=len(results),
            run_id=run.info.run_id,
        )

    def _build_gap_report(self, results: List[EvalResult]) -> GapReport:
        all_hallucinations: List[str] = []
        ood_failures: List[str] = []
        honesty_by_category: Dict[str, List[float]] = defaultdict(list)

        ood_refused = 0
        ood_attempted = 0

        for r in results:
            all_hallucinations.extend(r.hallucinated_details)
            honesty_by_category[r.input.category].append(
                r.scores.get("knowledge_honesty", 1.0)
            )

            if r.input.is_ood:
                intent_score = r.scores.get("intent_understanding", 0.0)
                if intent_score >= 0.7:
                    ood_refused += 1
                else:
                    ood_attempted += 1
                    ood_failures.append(r.input.text[:120] + "…")

        low_honesty = [
            cat for cat, scores in honesty_by_category.items()
            if (sum(scores) / max(len(scores), 1)) < 0.6
        ]

        in_dist = [r for r in results if not r.input.is_ood]
        avg_honesty = (
            sum(r.scores.get("knowledge_honesty", 1.0) for r in in_dist)
            / max(len(in_dist), 1)
        )

        return GapReport(
            hallucinated_details=list(set(all_hallucinations)),
            ood_pushback_failures=ood_failures,
            low_honesty_categories=low_honesty,
            ood_correctly_refused=ood_refused,
            ood_attempted_build=ood_attempted,
            avg_honesty_score=avg_honesty,
        )

    def get_best_run(self, node_name: str) -> Optional[dict]:
        runs = mlflow.search_runs(
            experiment_names=[self._experiment_name],
            filter_string=f"tags.node_name = '{node_name}'",
            order_by=["metrics.avg_overall_score DESC"],
            max_results=1,
        )
        return None if runs.empty else runs.iloc[0].to_dict()

    def get_history(self, node_name: str, limit: int = 50) -> list:
        runs = mlflow.search_runs(
            experiment_names=[self._experiment_name],
            filter_string=f"tags.node_name = '{node_name}'",
            order_by=["tags.iteration ASC"],
            max_results=limit,
        )
        if runs.empty:
            return []
        cols = [
            "tags.iteration", "metrics.avg_overall_score",
            "metrics.avg_knowledge_honesty", "tags.prompt_version", "run_id",
        ]
        available = [c for c in cols if c in runs.columns]
        return runs[available].to_dict("records")
