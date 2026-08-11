#!/usr/bin/env python3
"""Turn expts8_full.ini into deterministic dataset/shard task rows."""
import argparse, configparser, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True);p.add_argument("--shards",type=int,default=50);a=p.parse_args()
cfg=configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation());cfg.read(a.config)
rows=[]
for section in cfg.sections():
    if not section.startswith("testing.") or section=="testing.common" or "input_jsonl" not in cfg[section]:continue
    name=section.removeprefix("testing.");label=cfg[section].getint("expected_label")
    for shard in range(a.shards):rows.append({"dataset":name,"manifest":cfg[section]["input_jsonl"],"expected_label":label,"shard":shard,"shards":a.shards})
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text("".join(json.dumps(x,separators=(",",":"))+"\n" for x in rows));print(len(rows))
