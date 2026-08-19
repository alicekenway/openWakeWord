#!/usr/bin/env python3
"""Turn expts8_full.ini into deterministic dataset/shard task rows."""
import argparse, configparser, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True);p.add_argument("--groups-output",required=True);p.add_argument("--workers-per-test",type=int);a=p.parse_args()
cfg=configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation());cfg.read(a.config)
workers_per_test=a.workers_per_test or cfg.getint("slurm","workers_per_test",fallback=100)
sample_rate=cfg.getint("evaluation","sample_rate",fallback=16000)
window_seconds=cfg.getfloat("evaluation","audio_window_seconds",fallback=5.12)
stride_seconds=cfg.getfloat("evaluation","audio_window_stride_seconds",fallback=2.56)
window_samples=round(sample_rate*window_seconds);stride_samples=round(sample_rate*stride_seconds)
if workers_per_test < 1: raise SystemExit("workers_per_test must be >= 1")
if window_samples < 1 or stride_samples < 1 or stride_samples > window_samples: raise SystemExit("invalid evaluation window/stride")
rows=[];groups=[]
for section in cfg.sections():
    if not section.startswith("testing.") or section=="testing.common" or "input_jsonl" not in cfg[section]:continue
    name=section[len("testing."):];label=cfg[section].getint("expected_label");manifest=Path(cfg[section]["input_jsonl"])
    record_count=sum(bool(line.strip()) for line in manifest.open())
    workers=min(workers_per_test,record_count)
    start=len(rows)
    for worker in range(workers):rows.append({"dataset":name,"manifest":str(manifest),"expected_label":label,"shard":worker,"shards":workers,"window_samples":window_samples,"stride_samples":stride_samples})
    if workers:groups.append({"dataset":name,"start":start,"end":len(rows)-1,"workers":workers,"records":record_count})
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text("".join(json.dumps(x,separators=(",",":"))+"\n" for x in rows));print(len(rows))
Path(a.groups_output).write_text("".join(json.dumps(x,separators=(",",":"))+"\n" for x in groups))
