"""Command-line interface for the INI training pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .config import ConfigurationError, load_ini_config
from .runner import PipelineRunner
from .slurm import run_worker


def _force_steps(values: list[str] | None) -> set[str]:
    result: set[str] = set()
    for value in values or []:
        result.update(part.strip() for part in value.split(",") if part.strip())
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="INI-driven openWakeWord training pipeline")
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("run", help="Validate and run stages in [steps] order")
    run.add_argument("--config", required=True, help="INI configuration file")
    run.add_argument("--from", dest="from_step", help="Start at this listed stage")
    run.add_argument("--to", dest="to_step", help="Stop after this listed stage")
    run.add_argument("--force", action="append", help="Rerun a named stage; may be repeated or comma-separated")

    one = subcommands.add_parser("run-step", help="Run one named stage from [steps]")
    one.add_argument("--config", required=True, help="INI configuration file")
    one.add_argument("step", help="Stage name, for example feature.positive_train")
    one.add_argument("--force", action="store_true", help="Rerun even when its stage checkpoint is complete")

    validate = subcommands.add_parser("validate", help="Validate the full INI pipeline without running it")
    validate.add_argument("--config", required=True, help="INI configuration file")

    status = subcommands.add_parser("status", help="Show completion/staleness for each configured stage")
    status.add_argument("--config", required=True, help="INI configuration file")

    worker = subcommands.add_parser("__slurm-worker", help=argparse.SUPPRESS)
    worker.add_argument("--config", required=True, help=argparse.SUPPRESS)
    worker.add_argument("--config-root", required=True, help=argparse.SUPPRESS)
    worker.add_argument("--spec", required=True, help=argparse.SUPPRESS)
    worker.add_argument("--task-id", required=True, type=int, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "__slurm-worker":
            result = run_worker(
                config_path=args.config,
                config_root=args.config_root,
                spec_path=args.spec,
                task_id=args.task_id,
            )
            print(json.dumps(result, indent=2, sort_keys=True, default=str))
            return 0
        runner = PipelineRunner(load_ini_config(args.config))
        if args.command == "validate":
            plan = runner.validate()
            print("Configuration is valid. Planned stages:")
            for item in plan:
                print(f"- {item.name}")
            return 0
        if args.command == "status":
            print(json.dumps(runner.status(), indent=2, sort_keys=True))
            return 0
        if args.command == "run-step":
            result = runner.run(only_step=args.step, force_steps={args.step} if args.force else set())
        else:
            result = runner.run(
                from_step=args.from_step,
                to_step=args.to_step,
                force_steps=_force_steps(args.force),
            )
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    except (ConfigurationError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
