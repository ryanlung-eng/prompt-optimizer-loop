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
    structural_warnings: List[str]           # advisory graph-risk heuristics (Layer 2)
    soundness_issues: List[str]              # adversarial design/logic review findings (Layer 3)
    sound_rate: Optional[float]              # fraction of reviewed workflows with zero soundness issues


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
                "structural_warnings": r.structural.warnings,
                "soundness_reviewed": r.soundness_reviewed,
                "soundness_issues": r.soundness_issues,
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
            "structural_warnings": gap.structural_warnings,
            "soundness_issues": gap.soundness_issues,
            "sound_rate": gap.sound_rate,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(gap_dict, f, indent=2)
            tmp = f.name
        mlflow.log_artifact(tmp, artifact_path="gap_report")
        os.unlink(tmp)

        return IterationSummary(
            iteration=int(run.data.tags.get("iteration", 0)),
            node_name=run.data.tags.get("node_name", ""),
            prompt_version=run.data.tags.get("prompt_version", ""),
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
        all_structural_warnings: List[str] = []
        all_soundness_issues: List[str] = []

        ood_refused = 0
        ood_attempted = 0

        for r in results:
            all_hallucinations.extend(r.hallucinated_details)
            all_structural_warnings.extend(r.structural.warnings)
            all_soundness_issues.extend(r.soundness_issues)
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

        reviewed = [r for r in results if r.soundness_reviewed]
        sound_rate = (
            sum(1 for r in reviewed if not r.soundness_issues) / len(reviewed)
            if reviewed else None
        )

        return GapReport(
            hallucinated_details=list(set(all_hallucinations)),
            ood_pushback_failures=ood_failures,
            low_honesty_categories=low_honesty,
            ood_correctly_refused=ood_refused,
            ood_attempted_build=ood_attempted,
            avg_honesty_score=avg_honesty,
            structural_warnings=list(set(all_structural_warnings)),
            soundness_issues=list(set(all_soundness_issues)),
            sound_rate=sound_rate,
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


class HardBenchmarkTracer:
    """Logs one MLflow trace per (scenario, arm) pair from run_hard_benchmark,
    so every hard-scenario result's full detail — the conversation transcript,
    structural errors/warnings, Layer 3 soundness issues/blockers, and scores —
    lands in MLflow's Traces UI instead of existing only as console output that
    has to be pasted elsewhere to review. Deliberately a SEPARATE experiment
    from PromptTracker's (see BenchmarkConfig.trace_experiment_name) — these
    are shaped completely differently (one trace per scenario+arm, not one run
    per prompt iteration) and mixing them would make both harder to browse.

    Uses mlflow.start_span (stable since the tracing API's introduction) for
    everything that matters; mlflow.update_current_trace (added later, for
    trace-level searchable tags) is best-effort — every value it would have
    set is ALSO on the root span's own attributes, so an older installed
    mlflow still gets full trace content, just without the extra tag index.
    """

    def __init__(self, config: DatabricksConfig, experiment_name: str):
        os.environ["DATABRICKS_HOST"] = config.workspace_url
        os.environ["DATABRICKS_TOKEN"] = config.token
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment(experiment_name)

    def log_arm(self, arm: str, results: List[EvalResult]) -> None:
        for r in results:
            self._log_scenario(arm, r)

    def _log_scenario(self, arm: str, r: EvalResult) -> None:
        scenario = r.input.category
        with mlflow.start_span(name=f"{scenario}::{arm}") as root:
            root.set_inputs({
                "scenario": scenario,
                "prompt_text": r.input.text,
                "expected_behavior": r.input.expected_behavior,
            })

            # transcript always starts with a "user" entry and strictly
            # alternates user/ka (see WorkflowEvaluator._run_conversation) —
            # pair each user entry with the ka reply that follows it into one
            # child span per actual generation call, rather than one span
            # per raw message.
            transcript = r.transcript or []
            turn = 0
            i = 0
            while i < len(transcript):
                if transcript[i].get("role") != "user":
                    i += 1
                    continue
                user_entry = transcript[i]
                ka_entry = (
                    transcript[i + 1]
                    if i + 1 < len(transcript) and transcript[i + 1].get("role") == "ka"
                    else None
                )
                with mlflow.start_span(name=f"turn_{turn}") as t:
                    t.set_inputs({"prompt": user_entry.get("content", "")})
                    if ka_entry is not None:
                        t.set_outputs({"response": ka_entry.get("content", "")})
                turn += 1
                i += 2 if ka_entry is not None else 1

            root.set_outputs({
                "final_response": r.actual_response or "",
                "structural_valid": r.structural.valid,
                "structural_errors": r.structural.errors,
                "structural_warnings": r.structural.warnings,
                "soundness_issues": r.soundness_issues,
                "soundness_blockers": r.soundness_blockers,
                "soundness_would_approve": r.soundness_would_approve,
            })
            root.set_attributes({
                "arm": arm,
                "scenario": scenario,
                "weighted_score": r.weighted_score,
                "structural_valid": r.structural.valid,
                "blocker_count": len(r.soundness_blockers),
                "warning_count": len(r.structural.warnings),
                **{f"score_{k}": v for k, v in r.scores.items()},
            })
            try:
                mlflow.update_current_trace(tags={
                    "scenario": scenario,
                    "arm": arm,
                    "structural_valid": str(r.structural.valid),
                    "has_blockers": str(bool(r.soundness_blockers)),
                })
            except AttributeError:
                pass  # older mlflow — root span attributes above already carry this
