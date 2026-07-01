"""
Main orchestration loop.
Ties together synthetic data, evaluation, judging, tracking, and optimization.
"""
import asyncio
import hashlib
from typing import Dict, List

from rich.console import Console
from rich.table import Table

from .config import Config
from .evaluator import WorkflowEvaluator
from .judge import DatabricksJudge, EvalResult
from .optimizer import PromptOptimizer
from .synthetic_data import SyntheticInput, generate_dataset
from .tracker import PromptTracker

console = Console()


def _prompt_hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:7]


def _print_score_table(
    node_name: str, iteration: int, results: List[EvalResult], dim_names: List[str]
) -> float:
    in_dist = [r for r in results if not r.input.is_ood]
    ood = [r for r in results if r.input.is_ood]

    avg_overall = sum(r.weighted_score for r in results) / max(len(results), 1)
    dim_avgs = {
        d: sum(r.scores.get(d, 0.0) for r in results) / max(len(results), 1)
        for d in dim_names
    }

    table = Table(title=f"[bold]Node: {node_name} | Iteration {iteration}[/bold]")
    table.add_column("Metric", style="cyan")
    table.add_column("In-dist", justify="right")
    table.add_column("OOD", justify="right")
    table.add_column("Overall", justify="right")
    table.add_column("", justify="center")

    for dim in dim_names:
        ind_score = (
            sum(r.scores.get(dim, 0.0) for r in in_dist) / max(len(in_dist), 1)
            if in_dist else 0.0
        )
        ood_score = (
            sum(r.scores.get(dim, 0.0) for r in ood) / max(len(ood), 1)
            if ood else 0.0
        )
        overall_score = dim_avgs[dim]
        status = "✓" if overall_score >= 0.8 else "⚠" if overall_score >= 0.6 else "✗"
        table.add_row(dim, f"{ind_score:.3f}", f"{ood_score:.3f}", f"{overall_score:.3f}", status)

    table.add_section()
    status = "[green]PASS[/green]" if avg_overall >= 0.85 else "[red]FAIL[/red]"
    table.add_row(
        "[bold]OVERALL[/bold]", "", "", f"[bold]{avg_overall:.3f}[/bold]", status
    )

    # Deterministic structural check — did the KA actually produce valid n8n JSON,
    # independent of the LLM judge's subjective read of the response text.
    n = len(results)
    converged = sum(1 for r in results if r.structural.is_json)
    structurally_valid = sum(1 for r in results if r.structural.valid)
    table.add_row(
        "[dim]Converged to JSON[/dim]", "", "", f"[dim]{converged}/{n}[/dim]", ""
    )
    table.add_row(
        "[dim]Structurally valid[/dim]", "", "", f"[dim]{structurally_valid}/{n}[/dim]", ""
    )

    # OOD pushback summary row
    if ood:
        ood_refused = sum(1 for r in ood if r.scores.get("intent_understanding", 0) >= 0.7)
        table.add_row(
            "[dim]OOD pushback[/dim]", "", "",
            f"[dim]{ood_refused}/{len(ood)} correct[/dim]", ""
        )

    console.print(table)
    return avg_overall


def _print_gap_report(results: List[EvalResult]) -> None:
    """Print a concise knowledge-gap summary to help improve the knowledge base."""
    from collections import defaultdict

    ood = [r for r in results if r.input.is_ood]
    all_hallucinations = [h for r in results for h in r.hallucinated_details]
    low_honesty = [
        r for r in results
        if not r.input.is_ood and r.scores.get("knowledge_honesty", 1.0) < 0.6
    ]
    ood_attempted = [r for r in ood if r.scores.get("intent_understanding", 0) < 0.7]

    console.rule("[bold yellow]Knowledge Gap Report[/bold yellow]")

    if ood_attempted:
        console.print(f"\n[red]OOD requests the model tried to build ({len(ood_attempted)}):[/red]")
        for r in ood_attempted[:5]:
            console.print(f"  • {r.input.category}: {r.input.text[:100]}…")
            console.print(f"    [dim]→ {r.overall_comment}[/dim]")

    if all_hallucinations:
        unique_hallucinations = list(dict.fromkeys(all_hallucinations))  # dedupe, preserve order
        console.print(f"\n[red]Hallucinated details detected ({len(unique_hallucinations)} unique):[/red]")
        for h in unique_hallucinations[:10]:
            console.print(f"  • {h}")
        if len(unique_hallucinations) > 10:
            console.print(f"  … and {len(unique_hallucinations) - 10} more (see MLflow artifact)")

    if low_honesty:
        cats = list(dict.fromkeys(r.input.category for r in low_honesty))
        console.print(f"\n[yellow]Categories with low honesty scores:[/yellow] {', '.join(cats)}")

    structural_errors = [e for r in results for e in r.structural.errors]
    if structural_errors:
        unique_errors = list(dict.fromkeys(structural_errors))
        console.print(f"\n[red]Structural JSON errors detected ({len(unique_errors)} unique):[/red]")
        for e in unique_errors[:10]:
            console.print(f"  • {e}")
        if len(unique_errors) > 10:
            console.print(f"  … and {len(unique_errors) - 10} more (see MLflow artifact)")

    if not ood_attempted and not all_hallucinations and not low_honesty:
        console.print("[green]No significant knowledge gaps detected.[/green]")


def _print_native_assessments(results: List[EvalResult]) -> None:
    """
    Aggregate the Databricks-native judges (retrieval_groundedness, safety,
    Valid Workflow, etc. — whatever's configured on the KA endpoint) across
    this batch. Best-effort: those judges run as production-monitoring
    scorers and may not have finished computing when we fetched the trace —
    a case missing from the coverage count isn't necessarily a bad score,
    it may just not have been ready yet.
    """
    from collections import Counter, defaultdict

    by_name: dict = defaultdict(Counter)
    covered = 0
    for r in results:
        if r.native_assessments:
            covered += 1
        for name, value in r.native_assessments.items():
            by_name[str(name)][str(value)] += 1

    console.rule("[bold magenta]Native Databricks Judges[/bold magenta]")
    console.print(f"  Coverage: {covered}/{len(results)} examples had assessments attached at fetch time")

    if not by_name:
        console.print("[yellow]No native assessments found — either none are configured on this "
                       "endpoint, or none had finished computing yet.[/yellow]")
        return

    for name, counts in sorted(by_name.items()):
        total = sum(counts.values())
        breakdown = ", ".join(f"{v}: {n}/{total} ({n/total:.0%})" for v, n in counts.most_common())
        console.print(f"  [cyan]{name}[/cyan] — {breakdown}")


async def _evaluate_prompt(
    evaluator: WorkflowEvaluator,
    judge: DatabricksJudge,
    prompt: str,
    inputs: List[SyntheticInput],
) -> List[EvalResult]:
    """Run a prompt against all inputs, then judge all responses."""
    pairs = await evaluator.run_batch(prompt, inputs)
    results = await judge.evaluate_batch(pairs)
    return results


async def _optimize_node(
    node_name: str,
    current_prompt: str,
    iteration: int,
    inputs: List[SyntheticInput],
    evaluator: WorkflowEvaluator,
    judge: DatabricksJudge,
    optimizer: PromptOptimizer,
    tracker: PromptTracker,
    config: Config,
) -> tuple[str, float]:
    """
    Run one optimization cycle for a single node.
    Returns (best_prompt, best_score).
    """
    dim_names = [d.name for d in config.judge.dimensions]

    # --- Evaluate current prompt ---
    console.rule(f"[dim]Evaluating current prompt for '{node_name}'[/dim]")
    current_results = await _evaluate_prompt(evaluator, judge, current_prompt, inputs)
    current_score = _print_score_table(node_name, iteration, current_results, dim_names)

    with tracker.start_iteration(
        iteration=iteration,
        node_name=node_name,
        prompt_text=current_prompt,
        prompt_version=f"v{iteration}_{_prompt_hash(current_prompt)}",
        tags={"type": "baseline"},
    ) as run:
        tracker.log_results(run, current_results, dim_names)

    _print_gap_report(current_results)
    _print_native_assessments(current_results)

    if current_score >= config.optimizer.score_threshold:
        console.print(f"  [green]Score {current_score:.3f} ≥ threshold {config.optimizer.score_threshold}. No changes needed.[/green]")
        return current_prompt, current_score

    # --- Generate candidate prompts ---
    console.print(f"\n  [yellow]Score below threshold ({current_score:.3f}). Generating {config.optimizer.candidates_per_iteration} improved candidates…[/yellow]")
    candidates = await optimizer.generate_candidates(node_name, current_prompt, current_results)

    if not candidates:
        console.print("  [red]No candidates generated. Keeping current prompt.[/red]")
        return current_prompt, current_score

    # --- Evaluate each candidate ---
    best_prompt = current_prompt
    best_score = current_score

    for i, candidate in enumerate(candidates):
        version_tag = f"v{iteration}_candidate{i}_{_prompt_hash(candidate)}"
        console.print(f"\n  Testing candidate {i+1}/{len(candidates)} ({version_tag})…")

        candidate_results = await _evaluate_prompt(evaluator, judge, candidate, inputs)
        candidate_score = _print_score_table(
            node_name, iteration, candidate_results, dim_names
        )

        with tracker.start_iteration(
            iteration=iteration,
            node_name=node_name,
            prompt_text=candidate,
            prompt_version=version_tag,
            tags={"type": "candidate", "candidate_idx": str(i)},
        ) as run:
            tracker.log_results(run, candidate_results, dim_names)

        if candidate_score > best_score:
            best_score = candidate_score
            best_prompt = candidate
            console.print(f"  [green]↑ New best: {best_score:.3f}[/green]")

    improvement = best_score - current_score
    if improvement > 0:
        console.print(f"\n  [bold green]Best candidate improved score by +{improvement:.3f} → {best_score:.3f}[/bold green]")
    else:
        console.print(f"\n  [yellow]No candidate beat the current prompt. Keeping as-is.[/yellow]")

    return best_prompt, best_score


async def run_optimization_loop(
    config: Config,
    generate_only: bool = False,
    evaluate_only: bool = False,
):
    console.rule("[bold blue]n8n Prompt Optimizer[/bold blue]")
    console.print(f"  Nodes: {list(config.prompts.keys())}")
    console.print("  n8n write-back: disabled — copy the best prompt printed below into n8n manually")
    console.print()

    # --- Synthetic dataset ---
    console.rule("[dim]Synthetic dataset[/dim]")
    inputs = await generate_dataset(config.synthetic_data, config.databricks)
    ood_count = sum(1 for i in inputs if i.is_ood)
    console.print(f"  Dataset: {len(inputs)} inputs ({len(inputs) - ood_count} in-dist, {ood_count} OOD)")

    if generate_only:
        console.print("\n[green]--generate-only: done.[/green]")
        return

    current_prompts: Dict[str, str] = dict(config.prompts)
    if not current_prompts:
        console.print("[red]No prompts configured. Add entries under config.yaml's 'prompts' section.[/red]")
        return

    # --- Initialize services ---
    evaluator = WorkflowEvaluator(config.databricks)
    judge = DatabricksJudge(config.databricks, config.judge)
    optimizer = PromptOptimizer(config.optimizer, config.judge, config.databricks)
    tracker = PromptTracker(config.databricks)

    if evaluate_only:
        console.rule("[dim]Evaluate-only mode[/dim]")
        dim_names = [d.name for d in config.judge.dimensions]
        all_results = []
        for node_name, prompt in current_prompts.items():
            results = await _evaluate_prompt(evaluator, judge, prompt, inputs)
            _print_score_table(node_name, 0, results, dim_names)
            all_results.extend(results)
        _print_gap_report(all_results)
        _print_native_assessments(all_results)
        return

    # --- Optimization loop ---
    best_prompts = dict(current_prompts)

    for iteration in range(1, config.optimizer.max_iterations + 1):
        console.rule(f"[bold]Iteration {iteration}/{config.optimizer.max_iterations}[/bold]")

        all_converged = True
        updated_prompts: Dict[str, str] = {}

        for node_name, prompt in best_prompts.items():
            best_prompt, best_score = await _optimize_node(
                node_name=node_name,
                current_prompt=prompt,
                iteration=iteration,
                inputs=inputs,
                evaluator=evaluator,
                judge=judge,
                optimizer=optimizer,
                tracker=tracker,
                config=config,
            )
            updated_prompts[node_name] = best_prompt

            if best_score < config.optimizer.score_threshold:
                all_converged = False

        best_prompts = updated_prompts

        if all_converged:
            console.print(f"\n[bold green]All nodes converged at iteration {iteration}. Done![/bold green]")
            break
    else:
        console.print(f"\n[yellow]Reached max iterations ({config.optimizer.max_iterations}). "
                      f"Best prompts applied.[/yellow]")

    # --- Final summary ---
    console.rule("[bold]Best prompts — copy into n8n manually[/bold]")
    for node_name, prompt in best_prompts.items():
        history = tracker.get_history(node_name, limit=50)
        console.print(f"\n[bold]{node_name}[/bold]")
        console.print(f"  Iterations tracked: {len(history)}")
        if history:
            best_run = max(history, key=lambda r: r.get("metrics.avg_overall_score", 0))
            console.print(f"  Best MLflow run: {best_run.get('run_id', '')[:8]}… "
                          f"score={best_run.get('metrics.avg_overall_score', 0):.3f}")
        console.print(f"  Prompt hash: {_prompt_hash(prompt)}\n")
        console.rule(f"[dim]{node_name} — full prompt text[/dim]")
        console.print(prompt)
        console.rule()
