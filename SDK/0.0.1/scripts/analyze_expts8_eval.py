#!/usr/bin/env python3
"""Merge SDK shards and report one configured operating threshold."""
import argparse
import json
from collections import Counter
from pathlib import Path


def debounced_events(events, debounce_seconds):
    previous = -float("inf")
    count = 0
    for event in sorted(events, key=lambda value: float(value["end_sample"])):
        event_time = float(event["end_sample"]) / 16000.0
        if event_time - previous >= debounce_seconds:
            count += 1
            previous = event_time
    return count


def metric(value):
    return "n/a" if value is None else f"{float(value):.6f}"


parser = argparse.ArgumentParser()
parser.add_argument("--shards-root", required=True)
parser.add_argument("--keywords", required=True)
parser.add_argument("--model-dir", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--output-json")
parser.add_argument("--output-report")
args = parser.parse_args()

root = Path(args.shards_root)
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
keywords = json.loads(Path(args.keywords).read_text())["keywords"]
by_text = {value["display_text"].casefold(): value["id"] for value in keywords}
defaults = json.loads((Path(args.model_dir) / "sdk_defaults.json").read_text())
threshold = float(defaults["stage2_threshold"])
debounce_seconds = float(defaults.get("debounce_ms", 1000.0)) / 1000.0

summary = {
    "stage2_threshold": threshold,
    "debounce_seconds": debounce_seconds,
    "false_accept_rate_denominator": "evaluated_sliding_windows",
    "datasets": {},
    "total_records": 0,
    "total_errors": 0,
}
observed_windows = set()

for dataset in sorted(value for value in root.iterdir() if value.is_dir()):
    rows = []
    for part in sorted(dataset.glob("part-*.jsonl")):
        rows.extend(json.loads(line) for line in part.open() if line.strip())
    rows.sort(key=lambda value: value.get("source_index", -1))
    label = int((dataset / "expected_label").read_text())
    errors = sum("error" in row for row in rows)
    valid = [row for row in rows if "error" not in row]
    observed_windows.update((int(row.get("window_samples", 0)), int(row.get("stride_samples", 0))) for row in valid)
    window_count = sum(int(row.get("audio_window_count", 1)) for row in valid)
    audio_hours = sum(float(row.get("duration_seconds", row.get("samples", 0) / 16000.0)) for row in valid) / 3600.0
    event_count = sum(debounced_events(row.get("events", []), debounce_seconds) for row in valid)
    detected_sources = sum(bool(row.get("events")) for row in valid)
    detected_windows = sum(len({int(event.get("audio_window_index", 0)) for event in row.get("events", [])}) for row in valid)
    window_rtfs = [float(item["real_time_factor"]) for row in valid for item in row.get("inference_performance", []) if "real_time_factor" in item]
    rtfs = [float(row["rtf"]) for row in valid if "rtf" in row]
    values = {
        "expected_label": label,
        "records": len(rows),
        "evaluated_sources": len(valid),
        "evaluated_windows": window_count,
        "errors": errors,
        "audio_hours": audio_hours,
        "stage2_threshold": threshold,
        "debounced_event_count": event_count,
        "detected_sources": detected_sources,
        "processing_rtf_mean": sum(rtfs) / len(rtfs) if rtfs else None,
        "processing_rtf_max": max(rtfs) if rtfs else None,
        "window_rtf_mean": sum(window_rtfs) / len(window_rtfs) if window_rtfs else None,
        "window_rtf_max": max(window_rtfs) if window_rtfs else None,
    }
    if label:
        correct = 0
        wrong = 0
        confusion = Counter()
        for row in valid:
            expected = row.get("expected_keyword_id") or by_text.get(row.get("text", "").casefold(), "")
            detected = [event["keyword_id"] for event in row.get("events", [])]
            if expected and expected in detected:
                correct += 1
            elif detected:
                wrong += 1
                confusion[(expected, detected[0])] += 1
        false_rejects = len(valid) - correct
        values.update(
            correct=correct,
            wrong_keyword=wrong,
            false_reject=false_rejects,
            false_reject_rate=false_rejects / len(valid) if valid else None,
            recall=correct / len(valid) if valid else None,
            confusions=[{"expected": pair[0], "detected": pair[1], "count": count} for pair, count in confusion.most_common()],
        )
    else:
        values.update(
            false_accept_events=event_count,
            false_accept_sources=detected_sources,
            false_accept_windows=detected_windows,
            false_accepts_per_hour=event_count / audio_hours if audio_hours else None,
            false_accept_rate=detected_windows / window_count if window_count else None,
        )
    summary["datasets"][dataset.name] = values
    summary["total_records"] += len(rows)
    summary["total_errors"] += errors
    with (output / f"{dataset.name}.jsonl").open("w") as merged:
        for row in rows:
            merged.write(json.dumps(row, separators=(",", ":")) + "\n")

observed_windows.discard((0, 0))
if len(observed_windows) == 1:
    window_samples, stride_samples = next(iter(observed_windows))
    window_description = f"{window_samples / 16000.0:.6g} s window, {stride_samples / 16000.0:.6g} s stride"
else:
    window_description = "see per-record window_samples and stride_samples"
summary["sliding_window"] = window_description
report = [
    "# Wake-Word SDK Fixed-Threshold Summary",
    "",
    f"- Stage-2 threshold: `{threshold:.6g}`",
    f"- Debounce: `{debounce_seconds:.6g}` seconds",
    f"- Negative evaluation: sliding windows ({window_description})",
    "- FA rate: false-accepted sliding windows / evaluated sliding windows",
    "- FA/hour: debounced false-accept events / evaluated source-audio hours",
]
for name, values in summary["datasets"].items():
    report.extend([
        "", f"## {name}", "",
        f"- Evaluated source files: `{values['evaluated_sources']}`",
        f"- Evaluated sliding windows: `{values['evaluated_windows']}`",
        f"- Evaluated duration: `{values['audio_hours']:.6f}` hours",
        f"- Evaluation errors: `{values['errors']}`",
        "", "### Inference performance", "",
        f"- Measured windows: `{values['evaluated_windows']}`", "",
        "| Metric | Mean | Max |",
        "| --- | ---: | ---: |",
        f"| Real-time factor | {metric(values['window_rtf_mean'])} | {metric(values['window_rtf_max'])} |",
        "",
    ])
    if values["expected_label"]:
        report.extend([
            "| Stage-2 threshold | Accuracy / recall | False rejects | Wrong keyword | FR rate |",
            "| ---: | ---: | ---: | ---: | ---: |",
            f"| {threshold:.6g} | {metric(values['recall'])} | {values['false_reject']} | {values['wrong_keyword']} | {metric(values['false_reject_rate'])} |",
        ])
    else:
        report.extend([
            "| Stage-2 threshold | FA events | FA source files | FA windows | Evaluated windows | FA/hour | FA rate |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| {threshold:.6g} | {values['false_accept_events']} | {values['false_accept_sources']} | {values['false_accept_windows']} | {values['evaluated_windows']} | {metric(values['false_accepts_per_hour'])} | {metric(values['false_accept_rate'])} |",
        ])

json_path = Path(args.output_json) if args.output_json else output / "summary.json"
report_path = Path(args.output_report) if args.output_report else output / "FULL_CONDITION_COMPARISON.md"
json_path.parent.mkdir(parents=True, exist_ok=True)
report_path.parent.mkdir(parents=True, exist_ok=True)
json_path.write_text(json.dumps(summary, indent=2) + "\n")
report_path.write_text("\n".join(report).rstrip() + "\n")
print(json.dumps({"records": summary["total_records"], "errors": summary["total_errors"], "datasets": len(summary["datasets"]), "stage2_threshold": threshold}))
