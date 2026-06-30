#!/usr/bin/env python3
"""
n8n Prompt Optimizer — CLI entry point.

n8n is never read from or written to — paste the current prompt into
config.yaml's 'prompts' section, then copy the best-scoring prompt printed
at the end of the run back into n8n yourself.

Usage examples:
  python optimize.py                          # full loop with config.yaml
  python optimize.py --config custom.yaml
  python optimize.py --iterations 5
  python optimize.py --generate-only         # only generate + cache synthetic dataset
  python optimize.py --evaluate-only         # score current prompts, no optimization
  python optimize.py --clear-cache           # delete cached synthetic dataset then run
"""
import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--config", "config_path", default="config.yaml", show_default=True,
              help="Path to config.yaml")
@click.option("--iterations", type=int, default=None,
              help="Override max_iterations from config")
@click.option("--generate-only", is_flag=True, default=False,
              help="Only generate synthetic dataset (cached to disk) then exit")
@click.option("--evaluate-only", is_flag=True, default=False,
              help="Score current prompts without running the optimization loop")
@click.option("--clear-cache", is_flag=True, default=False,
              help="Delete cached synthetic dataset before running")
def main(config_path, iterations, generate_only, evaluate_only, clear_cache):
    from prompt_optimizer.config import load_config
    from prompt_optimizer.loop import run_optimization_loop

    try:
        cfg = load_config(config_path)
    except (FileNotFoundError, EnvironmentError) as e:
        console.print(f"[red]Config error:[/red] {e}")
        sys.exit(1)

    if iterations is not None:
        cfg.optimizer.max_iterations = iterations

    if clear_cache:
        cache = Path(cfg.synthetic_data.cache_path)
        if cache.exists():
            cache.unlink()
            console.print(f"Cleared synthetic dataset cache: {cache}")

    asyncio.run(run_optimization_loop(
        config=cfg,
        generate_only=generate_only,
        evaluate_only=evaluate_only,
    ))


if __name__ == "__main__":
    main()
