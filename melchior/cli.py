"""Command-line interface for Melchior."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from melchior.config import config
from melchior.brain.jev_client import JevClient
from melchior.brain.llm_client import LLMClient
from melchior.brain.decisions import assess_budget
from melchior.core.task import Task
from melchior.core.journal import Journal, TrialRecord
from melchior.core.strategist import Strategist
from melchior.core.filter import CandidateFilter
from melchior.core.executor import Executor
from melchior.core.report import generate_report


def run_experiment_loop(
    task_path: str,
    output_dir: str = "artifacts",
    n_candidates: int = 2,
) -> int:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    task = Task.from_yaml(task_path)
    journal = Journal(filepath=out_path / "journal.jsonl")

    jev = JevClient(
        mode=config.jev_mode,
        api_key=config.jev_api_key,
        api_url=config.jev_api_url,
    )
    llm = LLMClient(
        mode=config.llm_mode,
        model=config.llm_model,
        api_key=config.openai_api_key,
        base_url=config.llm_base_url,
    )

    strategist = Strategist(jev=jev, llm=llm)
    executor = Executor(jev=jev)
    c_filter = CandidateFilter(jev=jev, executor=executor)

    print(f"[*] Melchior initialized for task: '{task.name}'")
    print(f"[*] Domain: {task.domain} | Target metric: {task.metric} ({task.direction})")
    print(f"[*] System 1 (Jev): {jev.mode} | System 2 (LLM): {llm.mode} ({llm.model})")
    print(f"[*] Budget: {task.budget.max_trials} max trials / {task.budget.max_hours}h max time\n")

    trial_idx = len(journal.records)

    while journal.budget_remaining(task.budget):
        trial_idx += 1
        print(f"--- [Trial {trial_idx}/{task.budget.max_trials}] ---")

        # 1. Plan next step via dual-brain Strategist
        plan = strategist.plan_next_trial(task, journal, n_candidates=n_candidates)
        print(f"  [System 1 Jev] Selected Strategy: '{plan.strategy}'")
        print(f"  [System 1 Jev] Approved Techniques: {plan.viable_techniques}")
        print(f"  [System 2 LLM] Generated {len(plan.candidates)} candidate proposal(s)")

        # 2. Filter & rank via pilot runs
        best_so_far = journal.best_metric(task.direction)
        ranked = c_filter.filter_and_rank(
            plan.candidates,
            best_so_far=best_so_far if best_so_far != -float("inf") else 0.0,
            pilot_timeout_s=task.budget.pilot_timeout_s,
        )

        if not ranked:
            print("  [!] All candidates failed static checks or pilot runs. Retrying next trial...")
            continue

        selected_candidate, pilot_score = ranked[0]
        print(f"  [Filter] Selected top candidate: '{selected_candidate.technique}' (pilot score: {pilot_score:.3f})")
        print(f"  [Hypothesis] \"{selected_candidate.hypothesis}\"")

        # 3. Full execution with dense feedback & Jev early-stop
        print("  [Executor] Running full training...")
        exec_res = executor.run(
            code=selected_candidate.code,
            best_so_far=best_so_far if best_so_far != -float("inf") else 0.0,
            timeout_seconds=task.budget.trial_timeout_s,
        )

        actual_metric = exec_res.metric
        metric_display = f"{actual_metric:.4f}" if actual_metric is not None else "None"
        print(f"  [Result] Status: {exec_res.status} | Metric: {metric_display} (took {exec_res.wall_time_s:.2f}s)")

        # 4. Record trial to Journal
        record = TrialRecord(
            trial_idx=trial_idx,
            strategy=plan.strategy,
            technique=selected_candidate.technique,
            hypothesis=selected_candidate.hypothesis,
            code=selected_candidate.code,
            predicted_metric=selected_candidate.predicted_metric,
            actual_metric=actual_metric,
            status=exec_res.status,
            epoch_metrics=exec_res.epoch_metrics,
            wall_time_s=exec_res.wall_time_s,
            error_type=exec_res.error_type,
            error_message=exec_res.stderr[-200:] if exec_res.stderr else None,
        )
        journal.append(record)

        if record.calibration_error is not None:
            print(f"  [Calibration] Predicted {record.predicted_metric:.4f} vs Actual {record.actual_metric:.4f} (Δ: {record.calibration_error:+.4f})")

        # Check early target metric reached
        if task.target_metric is not None and actual_metric is not None:
            if (task.direction == "maximize" and actual_metric >= task.target_metric) or (
                task.direction == "minimize" and actual_metric <= task.target_metric
            ):
                print(f"\n[+] Target metric {task.target_metric} reached! Concluding research loop early.")
                break

        # 5. Check Jev budget assessment
        b_dec = assess_budget(jev, journal.summary())
        if not b_dec.should_continue:
            print(f"\n[!] Jev budget gate recommended early stopping (probability to gain further: {b_dec.probability:.2f})")
            break

        print()

    # Generate final report
    print("\n[*] Generating final research monograph and saving artifacts...")
    report_text = generate_report(journal, task, output_dir=out_path)
    best = journal.best_record(task.direction)
    best_str = f"{best.actual_metric:.4f} ({best.technique})" if best and best.actual_metric is not None else "None"
    print(f"[✓] Research complete. Best outcome: {best_str}")
    print(f"[✓] Saved report: {out_path / 'report.md'}")
    print(f"[✓] Saved code:   {out_path / 'best_solution.py'}")
    print(f"[✓] Saved ledger: {out_path / 'journal.jsonl'}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Melchior — Dual-Brain ML Auto-Research Agent")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run autonomous research loop on task.yaml")
    run_parser.add_argument("task", help="Path to task YAML specification file")
    run_parser.add_argument("--output", "-o", default="artifacts", help="Output directory for artifacts")
    run_parser.add_argument("--candidates", "-c", type=int, default=2, help="Candidates proposed per trial")

    report_parser = subparsers.add_parser("report", help="Generate report from existing journal")
    report_parser.add_argument("task", help="Path to task YAML specification file")
    report_parser.add_argument("--output", "-o", default="artifacts", help="Output directory")

    crucible_parser = subparsers.add_parser(
        "crucible", help="Run high-throughput execution-grounded self-play generator"
    )
    crucible_parser.add_argument(
        "--concurrency", "-j", type=int, default=20, help="Concurrent async workers (default: 20)"
    )
    crucible_parser.add_argument(
        "--output", "-o", default="data/crucible", help="Output directory for dataset jsonl files"
    )
    crucible_parser.add_argument(
        "--url", default=None, help="vLLM endpoint URL (default: http://localhost:8000/v1 or env CRUCIBLE_LLM_URL)"
    )
    crucible_parser.add_argument(
        "--model", default="qwen", help="Model name on vLLM server (default: qwen)"
    )
    crucible_parser.add_argument(
        "--mode", choices=["auto", "api", "opencode", "mock"], default="auto", help="Mode: auto | api | mock"
    )
    crucible_parser.add_argument(
        "--pairs", "-n", type=int, default=None, help="Attempt at most N pairs, including failures (default: run until stopped)"
    )
    crucible_parser.add_argument(
        "--timeout", type=float, default=12.0, help="Per-candidate timeout in seconds (default: 12.0)"
    )

    canary_parser = subparsers.add_parser(
        "canary", help="Run Canary Run (50-100 tasks) with OpenJev baseline benchmarking"
    )
    canary_parser.add_argument(
        "--tasks", "-n", type=int, default=50, help="Number of canary tasks to evaluate (default: 50)"
    )
    canary_parser.add_argument(
        "--concurrency", "-j", type=int, default=10, help="Concurrent workers (default: 10)"
    )
    canary_parser.add_argument(
        "--output", "-o", default="data/crucible_canary", help="Output directory for reports & holdout dataset"
    )
    canary_parser.add_argument(
        "--url", default="http://localhost:8000/v1", help="vLLM endpoint URL"
    )
    canary_parser.add_argument(
        "--model", default="qwen", help="LLM model name"
    )
    canary_parser.add_argument(
        "--jev-url", default="http://localhost:8080/v1/systemone", help="OpenJev endpoint URL"
    )
    canary_parser.add_argument(
        "--timeout", type=float, default=14.0, help="Sandbox timeout in seconds (default: 14.0)"
    )

    for run_parser in (crucible_parser, canary_parser):
        run_parser.add_argument("--reasoning-effort", choices=["low", "medium", "xhigh"], default="xhigh")
        run_parser.add_argument("--generation-timeout", type=float, default=600.)
        run_parser.add_argument("--generation-tokens", type=int, default=12288)
        run_parser.add_argument("--resume", action="store_true", help="Resume an identical run without duplicating completed seeds")
        run_parser.add_argument("--seed-start", type=int, default=1000)
        run_parser.add_argument("--max-hours", type=float, default=None, help="Stop after this session time budget; incomplete tasks can resume")
    crucible_parser.add_argument("--split", choices=["train", "eval"], default="train")

    args = parser.parse_args()

    if args.command == "run":
        sys.exit(run_experiment_loop(args.task, args.output, args.candidates))
    elif args.command == "report":
        task = Task.from_yaml(args.task)
        journal = Journal(Path(args.output) / "journal.jsonl")
        generate_report(journal, task, args.output)
        print(f"[✓] Generated report at {args.output}/report.md")
    elif args.command == "crucible":
        import asyncio
        from melchior.crucible.runner import CrucibleRunner

        runner = CrucibleRunner(
            concurrency=args.concurrency,
            reasoning_effort=args.reasoning_effort,
            generation_timeout_s=args.generation_timeout,
            generation_max_tokens=args.generation_tokens,
            output_dir=args.output,
            llm_base_url=args.url,
            llm_model=args.model,
            llm_mode=args.mode,
            sandbox_timeout_s=args.timeout,
            max_pairs=args.pairs,
            split=args.split, seed_start=args.seed_start, resume=args.resume, max_hours=args.max_hours,
        )
        asyncio.run(runner.run())
    elif args.command == "canary":
        import asyncio
        from melchior.crucible.canary import CanaryRunner

        runner = CanaryRunner(
            num_tasks=args.tasks,
            seed_start=args.seed_start, resume=args.resume, max_hours=args.max_hours,
            concurrency=args.concurrency,
            reasoning_effort=args.reasoning_effort,
            generation_timeout_s=args.generation_timeout,
            generation_max_tokens=args.generation_tokens,
            llm_url=args.url,
            llm_model=args.model,
            jev_url=args.jev_url,
            output_dir=args.output,
            sandbox_timeout_s=args.timeout,
        )
        asyncio.run(runner.run())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
