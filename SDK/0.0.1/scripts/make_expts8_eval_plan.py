#!/usr/bin/env python3
"""Turn expts8_full.ini into deterministic dataset/shard task rows."""
import argparse, configparser, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True);a=p.parse_args()
cfg=configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation());cfg.read(a.config)
rows=[]
for section in cfg.sections():
    if not section.startswith("testing.") or section=="testing.common" or "input_jsonl" not in cfg[section]:continue
    name=section[len("testing."):];label=cfg[section].getint("expected_label");manifest=Path(cfg[section]["input_jsonl"])
    record_count=sum(bool(line.strip()) for line in manifest.open())
    for record_index in range(record_count):rows.append({"dataset":name,"manifest":str(manifest),"expected_label":label,"shard":record_index,"shards":record_count})
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text("".join(json.dumps(x,separators=(",",":"))+"\n" for x in rows));print(len(rows))
