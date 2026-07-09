"""
Benchmarks the production Workflow Builder setup (engineered prompt + a
knowledge base + the KA's own endpoint) against a raw Claude endpoint, to
measure the actual value of this project rather than just iterate on it.

Three arms, all given the IDENTICAL system prompt and the SAME synthetic
inputs/judge/structural-validator as the regular eval loop — the only things
that vary are (a) which endpoint answers and (b) whether a knowledge base is
available to it:

  no_knowledge       — raw Sonnet (generation_endpoint), no KB access at all.
  knowledge_injected — raw Sonnet (generation_endpoint), the full flattened
                       KB corpus pasted directly into the prompt (~117k
                       tokens as of the current knowledge-base-upload/ set —
                       comfortably inside Sonnet's 200k context, so there's
                       no retrieval-quality confound: every fact is present
                       regardless of what the specific input needs).
  production         — the actual KA endpoint (eval_endpoint), whatever its
                       own internal knowledge access does.

Reading the three pairwise:
  no_knowledge vs knowledge_injected — isolates "does having accurate n8n
    syntax knowledge help at all," independent of prompt engineering (same
    prompt, same endpoint, same model — only KB access differs).
  knowledge_injected vs production — isolates "does the KA's own
    infrastructure add anything beyond just having the same text available
    in-context."
  no_knowledge vs production — the headline "value of this whole project"
    number.
"""
import asyncio
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.table import Table

from .config import Config
from .evaluator import WorkflowEvaluator
from .judge import DatabricksJudge, EvalResult
from .synthetic_data import SyntheticInput

console = Console()

_ARMS = ["no_knowledge", "knowledge_injected", "production"]
_ARM_LABELS = {
    "no_knowledge": "No KB (raw Sonnet)",
    "knowledge_injected": "KB injected (raw Sonnet)",
    "production": "Production (KA endpoint)",
}


def _load_knowledge_corpus(kb_path: str) -> str:
    """Concatenates every .md file in kb_path into one reference-docs blob."""
    kb_dir = Path(kb_path)
    if not kb_dir.is_dir():
        raise FileNotFoundError(
            f"Knowledge base directory not found: {kb_dir.resolve()} "
            f"(set benchmark.kb_path in config.yaml if it lives elsewhere)"
        )
    files = sorted(kb_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No .md files found in {kb_dir.resolve()}")
    parts = [f"# {f.stem}\n\n{f.read_text()}" for f in files]
    return "\n\n---\n\n".join(parts)


def _structural_validity_rate(results: List[EvalResult]) -> float:
    return sum(1 for r in results if r.structural.valid) / max(len(results), 1)


async def _run_arm(
    evaluator: WorkflowEvaluator,
    judge: DatabricksJudge,
    prompt: str,
    inputs: List[SyntheticInput],
    endpoint_url: str,
    use_responses_api: bool,
) -> List[EvalResult]:
    pairs = await evaluator.run_batch(
        prompt, inputs, endpoint_url=endpoint_url, use_responses_api=use_responses_api,
    )
    return await judge.evaluate_batch(pairs)


async def run_benchmark(
    config: Config,
    evaluator: WorkflowEvaluator,
    judge: DatabricksJudge,
    inputs: List[SyntheticInput],
) -> Dict[str, List[EvalResult]]:
    """
    Runs all three arms sequentially (each is internally concurrent via
    run_batch's own semaphore — running arms sequentially rather than nested
    keeps total concurrent load against Databricks endpoints predictable)
    and returns {arm_name: [EvalResult, ...]}.
    """
    base_prompt = config.prompts[config.benchmark.node_name]
    db = config.databricks

    raw_endpoint = f"{db.workspace_url}/serving-endpoints/{db.generation_endpoint}/invocations"
    ka_endpoint = f"{db.workspace_url}/serving-endpoints/{db.eval_endpoint}/invocations"

    kb_corpus = _load_knowledge_corpus(config.benchmark.kb_path)
    prompt_with_kb = (
        f"{base_prompt}\n\n---\n\nReference documentation (n8n node syntax, "
        f"gotchas, and patterns — use this as the authoritative source for "
        f"exact parameter names and node behavior, not just general "
        f"knowledge):\n\n{kb_corpus}"
    )

    results: Dict[str, List[EvalResult]] = {}

    console.print("  Running arm: no_knowledge (raw Sonnet, no KB access)…")
    results["no_knowledge"] = await _run_arm(
        evaluator, judge, base_prompt, inputs, raw_endpoint, use_responses_api=False,
    )

    console.print("  Running arm: knowledge_injected (raw Sonnet + full KB in-context)…")
    results["knowledge_injected"] = await _run_arm(
        evaluator, judge, prompt_with_kb, inputs, raw_endpoint, use_responses_api=False,
    )

    console.print("  Running arm: production (the actual KA endpoint)…")
    results["production"] = await _run_arm(
        evaluator, judge, base_prompt, inputs, ka_endpoint, use_responses_api=True,
    )

    return results


def print_benchmark_report(results: Dict[str, List[EvalResult]], dim_names: List[str]) -> None:
    table = Table(title="[bold]Benchmark: value of prompt engineering + knowledge base[/bold]")
    table.add_column("Metric", style="cyan")
    for arm in _ARMS:
        table.add_column(_ARM_LABELS[arm], justify="right")

    for dim in dim_names:
        row = [dim]
        for arm in _ARMS:
            r = results[arm]
            avg = sum(x.scores.get(dim, 0.0) for x in r) / max(len(r), 1)
            row.append(f"{avg:.3f}")
        table.add_row(*row)

    table.add_section()
    overall_row = ["OVERALL (weighted)"]
    for arm in _ARMS:
        r = results[arm]
        avg = sum(x.weighted_score for x in r) / max(len(r), 1)
        overall_row.append(f"{avg:.3f}")
    table.add_row(*overall_row)

    table.add_section()
    valid_row = ["Structurally valid"]
    for arm in _ARMS:
        r = results[arm]
        n = len(r)
        valid = sum(1 for x in r if x.structural.valid)
        valid_row.append(f"{valid}/{n} ({valid/max(n,1):.0%})")
    table.add_row(*valid_row)

    console.print(table)

    no_kb_valid = _structural_validity_rate(results["no_knowledge"])
    prod_valid = _structural_validity_rate(results["production"])
    console.print(
        f"\n[bold]Headline: {prod_valid:.0%} structurally valid with the full system "
        f"vs {no_kb_valid:.0%} with no knowledge base access at all "
        f"({(prod_valid - no_kb_valid):+.0%} points).[/bold]"
    )


def print_qualitative_examples(results: Dict[str, List[EvalResult]], n: int = 3) -> None:
    """
    Finds inputs where no_knowledge failed structurally but production
    succeeded on the exact same request, and prints the specific structural
    errors side by side — concrete "here's what base Claude got wrong"
    examples, since these tend to land harder than aggregate scores alone.
    """
    no_kb = {r.input.text: r for r in results["no_knowledge"]}
    prod = {r.input.text: r for r in results["production"]}

    candidates = [
        (text, no_kb[text], prod[text])
        for text in no_kb
        if text in prod and not no_kb[text].structural.valid and prod[text].structural.valid
    ]

    if not candidates:
        console.print("[yellow]No clean failure/success pairs found for qualitative examples "
                       "(either both arms succeeded, or both failed, on every shared input).[/yellow]")
        return

    console.rule("[bold]Concrete examples: no-KB failures the production system got right[/bold]")
    for text, no_kb_result, prod_result in candidates[:n]:
        console.print(f"\n[bold]Request:[/bold] {text[:150]}…")
        errors = "; ".join(no_kb_result.structural.errors[:3]) or "(no JSON produced at all)"
        console.print(f"[red]No KB — structural errors:[/red] {errors}")
        console.print(f"[green]Production — structurally valid:[/green] {prod_result.structural.valid}")


async def run(config: Config) -> Dict[str, List[EvalResult]]:
    """Entry point for the benchmark notebook."""
    from .synthetic_data import generate_dataset

    inputs = await generate_dataset(config.synthetic_data, config.databricks)
    evaluator = WorkflowEvaluator(config.databricks, cache_dir=str(Path(config.synthetic_data.cache_path).parent))
    judge = DatabricksJudge(config.databricks, config.judge)
    dim_names = [d.name for d in config.judge.dimensions]

    results = await run_benchmark(config, evaluator, judge, inputs)
    print_benchmark_report(results, dim_names)
    print_qualitative_examples(results)
    return results
