#!/usr/bin/env python3
"""Plan, run, inspect, and summarize SDK evaluation from one INI file."""
import argparse
import configparser
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


def command(values, *, capture=False):
    return subprocess.run(values, check=True, text=True, capture_output=capture)


def load_config(path):
    cfg = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    if not cfg.read(path):
        raise SystemExit(f"cannot read config: {path}")
    for section in ("sdk", "evaluation", "slurm", "summary"):
        if section not in cfg:
            raise SystemExit(f"missing [{section}] section")
    tests = [name for name in cfg.sections() if name.startswith("testing.") and name != "testing.common"]
    if not tests:
        raise SystemExit("no [testing.*] sections found")
    return cfg, tests


def paths(cfg):
    sdk_root = Path(__file__).resolve().parent.parent
    output = Path(cfg["sdk"]["output_dir"])
    return {
        "sdk_root": sdk_root,
        "model": Path(cfg["sdk"]["model_dir"]),
        "build": Path(cfg["sdk"]["build_dir"]),
        "output": output,
        "plan": output / "tasks.jsonl",
        "groups": output / "groups.jsonl",
        "logs": output / "logs",
        "jobs": output / "job_ids.jsonl",
        "merged": output / "merged",
    }


def validate(cfg, tests, value):
    for name, path in (("model_dir", value["model"]), ("build_dir", value["build"])):
        if not path.is_dir():
            raise SystemExit(f"{name} is not a directory: {path}")
    for required in (value["model"] / "keywords.json", value["model"] / "sdk_defaults.json", value["build"] / "wuw_eval_manifest"):
        if not required.is_file():
            raise SystemExit(f"missing required file: {required}")
    for section in tests:
        manifest = Path(cfg[section]["input_jsonl"])
        if not manifest.is_file():
            raise SystemExit(f"missing manifest for [{section}]: {manifest}")
        if cfg.getint(section, "expected_label") not in (0, 1):
            raise SystemExit(f"[{section}] expected_label must be 0 or 1")
    if cfg.getint("slurm", "workers_per_test") < 1:
        raise SystemExit("[slurm] workers_per_test must be >= 1")


def make_plan(config_path, cfg, tests, value):
    value["output"].mkdir(parents=True, exist_ok=True)
    value["logs"].mkdir(parents=True, exist_ok=True)
    command([sys.executable, str(value["sdk_root"] / "scripts/make_expts8_eval_plan.py"), "--config", str(config_path), "--output", str(value["plan"]), "--groups-output", str(value["groups"])])
    groups = [json.loads(line) for line in value["groups"].open() if line.strip()]
    print(f"Planned {sum(item['workers'] for item in groups)} workers for {sum(item['records'] for item in groups)} audio files across {len(groups)} test sets.")
    for item in groups:
        per_worker = (item["records"] + item["workers"] - 1) // item["workers"]
        print(f"  {item['dataset']}: {item['records']} files -> {item['workers']} workers (up to {per_worker} files/worker)")
    return groups


def wait_for(job, poll_seconds):
    while command(["squeue", "-h", "-j", job], capture=True).stdout.strip():
        time.sleep(poll_seconds)
    states = command(["sacct", "-n", "-X", "-j", job, "--format=State"], capture=True).stdout
    if any(bad in states for bad in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY")):
        raise SystemExit(f"Slurm job {job} failed; inspect its logs")


def summarize(cfg, value):
    args = [sys.executable, str(value["sdk_root"] / "scripts/analyze_expts8_eval.py"), "--shards-root", str(value["output"] / "shards"), "--keywords", str(value["model"] / "keywords.json"), "--model-dir", str(value["model"]), "--output", str(value["merged"])]
    if cfg["summary"].get("output_json"):
        args += ["--output-json", cfg["summary"]["output_json"]]
    if cfg["summary"].get("output_report"):
        args += ["--output-report", cfg["summary"]["output_report"]]
    command(args)


def run(config_path, cfg, tests, value):
    groups = make_plan(config_path, cfg, tests, value)
    if (value["output"] / "shards").exists():
        raise SystemExit(f"output already contains shards; choose a fresh [sdk] output_dir: {value['output']}")
    sbatch_args = shlex.split(cfg["slurm"].get("sbatch_args", "--partition=cpu --cpus-per-task=1 --mem=2G --time=02:00:00"))
    max_running = cfg.getint("slurm", "max_concurrent_per_test", fallback=cfg.getint("slurm", "workers_per_test"))
    poll = cfg.getint("slurm", "poll_seconds", fallback=20)
    value["jobs"].write_text("")
    for group in groups:
        limit = min(group["workers"], max_running)
        array = f"{group['start']}-{group['end']}%{limit}"
        result = command(["sbatch", "--parsable", *sbatch_args, f"--array={array}", f"--output={value['logs']}/%A_%a.out", f"--error={value['logs']}/%A_%a.err", str(value["sdk_root"] / "scripts/expts8_sbatch_entry.sh"), str(value["sdk_root"] / "scripts/expts8_array_worker.sh"), str(value["plan"]), str(value["model"]), str(value["build"]), str(value["output"])], capture=True)
        job = result.stdout.strip().split(";")[0]
        with value["jobs"].open("a") as stream:
            stream.write(json.dumps({"dataset": group["dataset"], "job_id": job}) + "\n")
        print(f"Submitted {group['dataset']}: job {job}, {group['workers']} workers")
        wait_for(job, poll)
    expected = sum(group["workers"] for group in groups)
    completed = sum(1 for _ in (value["output"] / "shards").glob("*/part-*.jsonl"))
    if completed != expected:
        raise SystemExit(f"expected {expected} shard files, found {completed}")
    if cfg.getboolean("summary", "enabled", fallback=True):
        summarize(cfg, value)


def status(value):
    if not value["jobs"].is_file():
        raise SystemExit(f"no job file: {value['jobs']}")
    for row in (json.loads(line) for line in value["jobs"].open() if line.strip()):
        state = command(["sacct", "-n", "-X", "-j", row["job_id"], "--format=State"], capture=True).stdout.strip().splitlines()
        print(f"{row['dataset']}\t{row['job_id']}\t{state[0].strip() if state else 'UNKNOWN'}")


parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("action", choices=("plan", "run", "status", "summary"), nargs="?", default="plan")
args = parser.parse_args()
config_path = Path(args.config).resolve()
config, test_sections = load_config(config_path)
value = paths(config)
validate(config, test_sections, value)
if args.action == "plan": make_plan(config_path, config, test_sections, value)
elif args.action == "run": run(config_path, config, test_sections, value)
elif args.action == "status": status(value)
else: summarize(config, value)
