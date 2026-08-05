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
from .state_simulation import STATE_SIMULATION_INSTRUCTION
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


# --------------------------------------------------------------------- #
# Hard-scenario benchmark: custom RAG pipeline variants (the IDENTICAL    #
# production prompt from config.yaml, plus retrieval over                #
# knowledge-base-upload/ spliced in) — a harder, more realistic          #
# comparison than the 3-arm one above, since it uses the Layer 4         #
# hand-crafted scenarios (self-loops, unwired sub-nodes, multi-hop       #
# chains) instead of the easier trigger×output synthetic set.           #
#                                                                        #
# Production is deliberately NOT one of these arms — across every run   #
# so far the custom RAG pipeline has consistently and clearly exceeded  #
# it, so this now isolates differences BETWEEN the custom pipeline      #
# variants themselves: v1 (top-K retrieval only) vs v2 (+ relevance     #
# filter + grounding note) vs v2 + the execution-trace checker (a       #
# narrow, cheap-model second opinion wired into the self-repair loop     #
# itself — see execution_checker.py — not just a post-hoc measurement). #
# --------------------------------------------------------------------- #

# custom_rag_v2_checked (the in-loop execution-trace checker) was dropped
# after three benchmark runs: it consistently scored WORST overall despite
# having the fewest blockers, because its repair turns trade completeness for
# blocker-avoidance, and trace analysis showed every one of its repair turns
# issuing at least one false demand (a nonexistent double-wrapped output
# path, re-routing the approval DM away from the workflow owner). The
# execution_check=True toggle still exists in evaluator.py if it's ever worth
# revisiting with a stronger checker model.
# custom_rag_v2_rewritten (grounded query rewriting) was dropped after the
# requirement-coverage review landed: under the earlier, more lenient
# reviewer it looked best (fewest blockers), but once the reviewer actually
# hunted for silently-dropped requirements the ranking inverted — it became
# the WORST arm (19 blockers vs plain v2's 12, completeness 0.885 vs 0.903).
# Its apparent advantage was avoiding runtime errors while dropping more of
# what the user asked for, which the old reviewer wasn't looking for. It also
# costs an extra vector search + Haiku call per scenario. The query_rewrite=True
# toggle remains in evaluator.py if it's ever worth revisiting.
# custom_rag (v1: plain top-K retrieval, no relevance filter) was dropped as
# a standing arm — it lost consistently across every run, and with the
# interesting comparisons now all v2-relative it was only costing runtime.
# evaluator.run_batch_custom_rag still exists if it's ever worth re-checking.
_HARD_ARMS = ["custom_rag_v2", "custom_rag_v2_statesim", "custom_rag_v2_strong"]
_HARD_ARM_LABELS = {
    "custom_rag_v2": "Custom RAG v2 (+ relevance filter + grounding note)",
    "custom_rag_v2_statesim": "Custom RAG v2 + state-transition self-simulation",
    "custom_rag_v2_strong": "Custom RAG v2 on a stronger generation model",
}

# Every string in StructuralResult.warnings comes from exactly one of these
# validator.py checks (confirmed by reading validator.py directly, not
# guessed from message content) — matched here on the fixed substring each
# one always emits, so a category count is exact, not a fuzzy keyword guess.
_LAYER2_WARNING_CATEGORIES = [
    ("typeVersion possibly stale", lambda w: "is not a real version of that node" in w),
    ("approval gate missing", lambda w: "no approval gate upstream" in w),
    ("possible self-loop risk", lambda w: w.startswith("Possible infinite-loop risk")),
    ("placeholder identity guard", lambda w: "silently disabling the loop protection" in w),
    ("non-tool node wired as ai_tool", lambda w: "n8n will not expose it to the agent" in w),
    ("output parser disabled", lambda w: "n8n silently ignores the connected parser" in w),
]


async def run_hard_benchmark(
    config: Config,
    evaluator: WorkflowEvaluator,
    judge: DatabricksJudge,
) -> Dict[str, List[EvalResult]]:
    from dataclasses import replace as _dc_replace

    from .hard_scenarios import load_hard_scenarios

    inputs = load_hard_scenarios()
    base_prompt = config.prompts[config.benchmark.node_name]

    # Same prompt text production receives (see run_batch_custom_rag's seam-
    # splice branch) — NOT instructions.md, so every arm isolates retrieval/
    # pipeline differences rather than also confounding them with a
    # different rules document. Production itself is no longer benchmarked
    # here — the custom RAG arms have consistently and clearly exceeded it,
    # so this now isolates differences BETWEEN the custom pipeline variants
    # instead (v1 vs v2 vs v2+execution-checker).
    base_instructions = base_prompt

    # Override enabled=True regardless of config.yaml's rag.enabled — running
    # this benchmark IS the point, independent of whether the main eval loop
    # has the pipeline switched on yet. dataclasses.replace (rather than
    # rebuilding field-by-field) so every RAGConfig field config.yaml exposes
    # — including generation_endpoint/max_chunks_per_source/
    # over_fetch_multiplier, previously NOT forwarded here at all, so
    # overriding them in config.yaml was silently ignored by this benchmark —
    # actually reaches both custom-RAG arms below.
    rag_config = _dc_replace(config.rag, enabled=True)

    results: Dict[str, List[EvalResult]] = {}

    # Optional — logs one MLflow trace per (scenario, arm) with the full
    # transcript, structural errors/warnings, and soundness issues/blockers,
    # so a failure can be drilled into in MLflow instead of only existing as
    # console output. Tracing is best-effort: a failure here (auth hiccup,
    # mlflow API mismatch) must never take down the benchmark run itself,
    # since the report printed at the end is the thing this script exists for.
    tracer = None
    if config.benchmark.trace_experiment_name:
        try:
            from .tracker import HardBenchmarkTracer
            tracer = HardBenchmarkTracer(config.databricks, config.benchmark.trace_experiment_name)
        except Exception as e:
            console.print(f"  [yellow]Warning: could not initialize hard-benchmark tracing "
                          f"({e}) — continuing without it.[/yellow]")

    def _trace(arm: str, arm_results: List[EvalResult]) -> None:
        if tracer is None:
            return
        try:
            tracer.log_arm(arm, arm_results)
        except Exception as e:
            console.print(f"  [yellow]Warning: failed to log traces for arm '{arm}' ({e}).[/yellow]")

    console.print(f"  Running arm: custom_rag_v2 (relevance filter + grounding) "
                  f"on {len(inputs)} hard scenarios…")
    pairs_v2 = await evaluator.run_batch_custom_rag_v2(base_instructions, inputs, rag_config)
    results["custom_rag_v2"] = await judge.evaluate_batch(pairs_v2)
    _trace("custom_rag_v2", results["custom_rag_v2"])

    console.print(f"  Running arm: custom_rag_v2_statesim (+ state-transition "
                  f"self-simulation) on {len(inputs)} hard scenarios…")
    pairs_statesim = await evaluator.run_batch_custom_rag_v2(
        base_instructions, inputs, rag_config,
        extra_instructions=STATE_SIMULATION_INSTRUCTION,
    )
    results["custom_rag_v2_statesim"] = await judge.evaluate_batch(pairs_statesim)
    _trace("custom_rag_v2_statesim", results["custom_rag_v2_statesim"])

    # Same pipeline and same prompt as plain custom_rag_v2 — ONLY the
    # generation model differs, so this arm answers "how much of what's left
    # is a model ceiling rather than a scaffolding gap?" The relevance
    # filter and judge stay on their usual endpoints deliberately; moving
    # them too would confound the comparison. The endpoint already forms
    # part of the conversation cache key, so no extra cache handling is
    # needed. If the endpoint name is wrong, per-scenario failures surface
    # as [ERROR: ...] results for THIS arm only rather than killing the run.
    strong_endpoint = config.benchmark.strong_model_endpoint
    if strong_endpoint:
        _delay = config.benchmark.strong_model_request_delay
        console.print(f"  Running arm: custom_rag_v2_strong ({strong_endpoint}) on "
                      f"{len(inputs)} hard scenarios — serialized with a {_delay}s "
                      f"pause per call for the TPM quota, so this arm is SLOW…")
        # Serialized (max_concurrent=1). The first attempt at this arm failed
        # ALL 17 scenarios with 429 REQUEST_LIMIT_EXCEEDED — a workspace
        # tokens-per-minute quota on Opus 5, not a bad endpoint name. These
        # prompts carry ~25k chars of retrieved context each, so four
        # concurrent multi-turn conversations blow an input-TPM cap
        # instantly and then never recover. One at a time is slow but
        # actually completes; raising the workspace quota is the real fix.
        pairs_strong = await evaluator.run_batch_custom_rag_v2(
            base_instructions, inputs,
            _dc_replace(rag_config, generation_endpoint=strong_endpoint),
            max_concurrent=1,
            request_delay=config.benchmark.strong_model_request_delay,
        )
        results["custom_rag_v2_strong"] = await judge.evaluate_batch(pairs_strong)
        _trace("custom_rag_v2_strong", results["custom_rag_v2_strong"])
    else:
        console.print("  Skipping arm: custom_rag_v2_strong "
                      "(benchmark.strong_model_endpoint not set)")
        results["custom_rag_v2_strong"] = []

    if tracer is not None:
        console.print(f"  Traces logged to MLflow experiment: {config.benchmark.trace_experiment_name}")

    return results


def print_hard_benchmark_report(results: Dict[str, List[EvalResult]], dim_names: List[str]) -> None:
    table = Table(title="[bold]Hard-scenario benchmark: custom RAG pipeline variants[/bold]")
    table.add_column("Metric", style="cyan")
    for arm in _HARD_ARMS:
        table.add_column(_HARD_ARM_LABELS[arm], justify="right")

    for dim in dim_names:
        row = [dim]
        for arm in _HARD_ARMS:
            r = results[arm]
            avg = sum(x.scores.get(dim, 0.0) for x in r) / max(len(r), 1)
            row.append(f"{avg:.3f}")
        table.add_row(*row)

    table.add_section()
    overall_row = ["OVERALL (weighted)"]
    for arm in _HARD_ARMS:
        r = results[arm]
        avg = sum(x.weighted_score for x in r) / max(len(r), 1)
        overall_row.append(f"{avg:.3f}")
    table.add_row(*overall_row)

    table.add_section()
    valid_row = ["Structurally valid"]
    for arm in _HARD_ARMS:
        r = results[arm]
        n = len(r)
        valid = sum(1 for x in r if x.structural.valid)
        valid_row.append(f"{valid}/{n} ({valid/max(n, 1):.0%})")
    table.add_row(*valid_row)

    # These two rows are the actual point of the hard dataset — self-loop
    # risk and design/logic soundness, not just "does it parse."
    table.add_section()
    warn_row = ["Advisory warnings (Layer 2, total)"]
    for arm in _HARD_ARMS:
        warn_row.append(str(sum(len(x.structural.warnings) for x in results[arm])))
    table.add_row(*warn_row)

    # Broken down by category, since the total alone reads as a quality
    # signal but isn't one: these three checks fire on SURFACE AREA (a
    # workflow with more outbound-send nodes or more trigger+send pairs has
    # more opportunities to trip "no approval gate"/"self-loop risk" purely
    # by having more of the thing being checked), not on defect severity.
    # An arm scoring worse on blockers can still show fewer warnings simply
    # by omitting the gated behavior entirely rather than getting it right.
    for label, matcher in _LAYER2_WARNING_CATEGORIES:
        row = [f"  - {label}"]
        for arm in _HARD_ARMS:
            row.append(str(sum(
                1 for x in results[arm] for w in x.structural.warnings if matcher(w)
            )))
        table.add_row(*row)

    # Catches drift if validator.py grows a 4th warnings-producing check
    # without this list being updated — should read 0 for every arm; a
    # nonzero value here means some warning text no longer matches any
    # known category and the breakdown above is silently incomplete.
    other_row = ["  - other/uncategorized"]
    for arm in _HARD_ARMS:
        other_row.append(str(sum(
            1 for x in results[arm] for w in x.structural.warnings
            if not any(matcher(w) for _label, matcher in _LAYER2_WARNING_CATEGORIES)
        )))
    table.add_row(*other_row)

    # Label reads "N/M reviewed" rather than embedding a bare fraction, since
    # "0 soundness issues" in a column header previously read as if 0 were a
    # fixed count rather than the qualifying criterion (zero issues = sound).
    # M (the denominator) is NOT the arm's full scenario count — it's only
    # calls that produced something resembling JSON at all (soundness_reviewed
    # requires a `{`...`}` span in the response); an arm with a lot of outright
    # failures (HTTP errors, pure-prose non-attempts) will show a smaller M,
    # which is itself a signal, not a discrepancy to reconcile against N.
    # Headline soundness signal. "Zero issues of any severity" (kept below for
    # continuity) read 0/N for every arm in four consecutive runs, because the
    # reviewer lists 3-7 findings per workflow and scores "no retry on this
    # HTTP call" identically to "infinite loop" — so it could not separate arms
    # the judge score clearly does. Blockers are the subset that mean the
    # workflow will not do what was asked.
    blocker_row = ["Blocker-free (no severity=blocker)"]
    for arm in _HARD_ARMS:
        reviewed = [x for x in results[arm] if x.soundness_reviewed]
        clean = sum(1 for x in reviewed if not x.soundness_blockers)
        blocker_row.append(f"{clean}/{len(reviewed)} reviewed" if reviewed else "n/a")
    table.add_row(*blocker_row)

    total_blockers = ["Blockers (total)"]
    for arm in _HARD_ARMS:
        total_blockers.append(str(sum(len(x.soundness_blockers) for x in results[arm])))
    table.add_row(*total_blockers)

    sound_row = ["Sound (zero issues, any severity)"]
    for arm in _HARD_ARMS:
        reviewed = [x for x in results[arm] if x.soundness_reviewed]
        sound = sum(1 for x in reviewed if not x.soundness_issues)
        sound_row.append(f"{sound}/{len(reviewed)} reviewed" if reviewed else "n/a")
    table.add_row(*sound_row)

    # "Zero issues" is an extremely strict bar — the reviewer mixes genuine
    # blockers (infinite loop, broken data flow) with pedantic-but-true nits
    # ("no retry on this HTTP call"), so a shippable workflow can still never
    # score as sound, and the row above reads 0/N for every arm regardless of
    # real differences between them. This is the same review's own ship/no-ship
    # verdict, which does discriminate.
    approve_row = ["Reviewer would approve"]
    for arm in _HARD_ARMS:
        verdicts = [x for x in results[arm] if x.soundness_would_approve is not None]
        approved = sum(1 for x in verdicts if x.soundness_would_approve)
        approve_row.append(f"{approved}/{len(verdicts)} reviewed" if verdicts else "n/a")
    table.add_row(*approve_row)

    # Hallucinated model-provider check: OpenAI is NOT an available credential
    # on this platform (config.yaml lists Databricks/Gmail/Sheets/Jira/Slack/
    # Docs/Drive/Slides/Calendar only), but nothing in Layers 1-3 currently
    # fails a workflow for using it, so it stayed invisible while the arm doing
    # it most scored HIGHEST on knowledge_honesty. Surfaced as a report row
    # rather than a hard validator error so it doesn't silently change the
    # structural-validity metric mid-comparison.
    unavailable_row = ["Uses unavailable provider (OpenAI)"]
    for arm in _HARD_ARMS:
        hits = sum(
            1 for x in results[arm]
            if any(m in (x.actual_response or "").lower()
                   for m in ("lmchatopenai", "nodes-langchain.openai", "nodes-base.openai", "gpt-4"))
        )
        unavailable_row.append(str(hits))
    table.add_row(*unavailable_row)

    console.print(table)


def print_qualitative_examples_hard(
    results: Dict[str, List[EvalResult]], arm_a: str = "custom_rag_v2",
    arm_b: str = "custom_rag_v2_statesim", n: int = 5,
) -> None:
    """
    With only 17 scenarios, every disagreement is worth seeing directly —
    unlike print_qualitative_examples (which only checks one direction to
    build a monotonic "value of this project" story), this checks both
    directions between any two arms. Defaults to custom_rag vs custom_rag_v2;
    pass arm_a/arm_b to compare any other pair (e.g. custom_rag_v2 vs
    custom_rag, to see concretely what the v2 additions changed).
    """
    a_results = {r.input.text: r for r in results[arm_a]}
    b_results = {r.input.text: r for r in results[arm_b]}
    a_label = _HARD_ARM_LABELS.get(arm_a, arm_a)
    b_label = _HARD_ARM_LABELS.get(arm_b, arm_b)

    b_wins = [
        (text, a_results[text], b_results[text]) for text in a_results
        if text in b_results and not a_results[text].structural.valid
        and b_results[text].structural.valid
    ]
    a_wins = [
        (text, a_results[text], b_results[text]) for text in a_results
        if text in b_results and a_results[text].structural.valid
        and not b_results[text].structural.valid
    ]

    if b_wins:
        console.rule(f"[bold green]{b_label} got right, {a_label} got wrong[/bold green]")
        for text, a_r, b_r in b_wins[:n]:
            console.print(f"\n[bold]Scenario:[/bold] {text[:150]}…")
            errors = "; ".join(a_r.structural.errors[:3]) or "(no JSON produced at all)"
            console.print(f"[red]{a_label} — structural errors:[/red] {errors}")
            console.print(f"[green]{b_label} — structurally valid:[/green] {b_r.structural.valid}")

    if a_wins:
        console.rule(f"[bold yellow]{a_label} got right, {b_label} got wrong[/bold yellow]")
        for text, a_r, b_r in a_wins[:n]:
            console.print(f"\n[bold]Scenario:[/bold] {text[:150]}…")
            errors = "; ".join(b_r.structural.errors[:3]) or "(no JSON produced at all)"
            console.print(f"[red]{b_label} — structural errors:[/red] {errors}")
            console.print(f"[green]{a_label} — structurally valid:[/green] {a_r.structural.valid}")

    if not b_wins and not a_wins:
        console.print(f"[yellow]No clean failure/success disagreements between {a_label} and "
                       f"{b_label} — either both succeeded or both failed on every shared scenario.[/yellow]")


async def run_hard(config: Config) -> Dict[str, List[EvalResult]]:
    """Entry point for the hard-scenario benchmark notebook."""
    evaluator = WorkflowEvaluator(config.databricks, cache_dir=str(Path(config.synthetic_data.cache_path).parent))
    judge = DatabricksJudge(config.databricks, config.judge)
    dim_names = [d.name for d in config.judge.dimensions]

    results = await run_hard_benchmark(config, evaluator, judge)
    print_hard_benchmark_report(results, dim_names)
    console.rule("[bold]custom_rag_v2 vs custom_rag_v2_statesim (state simulation)[/bold]")
    print_qualitative_examples_hard(results, "custom_rag_v2", "custom_rag_v2_statesim")
    if results.get("custom_rag_v2_strong"):
        console.rule("[bold]custom_rag_v2 vs custom_rag_v2_strong (stronger model)[/bold]")
        print_qualitative_examples_hard(results, "custom_rag_v2", "custom_rag_v2_strong")
    return results
