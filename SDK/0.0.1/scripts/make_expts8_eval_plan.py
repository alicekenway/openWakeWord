#!/usr/bin/env python3
"""Turn expts8_full.ini into deterministic dataset/shard task rows."""
import argparse, configparser, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True);p.add_argument("--groups-output",required=True);p.add_argument("--workers-per-test",type=int,default=100);a=p.parse_args()
if a.workers_per_test < 1: raise SystemExit("--workers-per-test must be >= 1")
cfg=configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation());cfg.read(a.config)
rows=[];groups=[]
for section in cfg.sections():
    if not section.startswith("testing.") or section=="testing.common" or "input_jsonl" not in cfg[section]:continue
    name=section[len("testing."):];label=cfg[section].getint("expected_label");manifest=Path(cfg[section]["input_jsonl"])
    record_count=sum(bool(line.strip()) for line in manifest.open())
    workers=min(a.workers_per_test,record_count)
    start=len(rows)
    for worker in range(workers):rows.append({"dataset":name,"manifest":str(manifest),"expected_label":label,"shard":worker,"shards":workers})
    if workers:groups.append({"dataset":name,"start":start,"end":len(rows)-1,"workers":workers,"records":record_count})
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text("".join(json.dumps(x,separators=(",",":"))+"\n" for x in rows));print(len(rows))
Path(a.groups_output).write_text("".join(json.dumps(x,separators=(",",":"))+"\n" for x in groups))
