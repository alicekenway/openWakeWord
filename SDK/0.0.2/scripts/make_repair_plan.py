#!/usr/bin/env python3
"""Select only dataset/shard tasks whose current result contains an error."""
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--tasks",required=True);p.add_argument("--shards-root",required=True);p.add_argument("--output",required=True);a=p.parse_args()
tasks=[json.loads(line) for line in open(a.tasks) if line.strip()];needed=set();root=Path(a.shards_root)
for path in root.glob("*/part-*.jsonl"):
    if '"error"' in path.read_text():needed.add((path.parent.name,int(path.stem.split("-")[-1])))
selected=[x for x in tasks if (x["dataset"],x["shard"]) in needed]
Path(a.output).write_text("".join(json.dumps(x,separators=(",",":"))+"\n" for x in selected));print(len(selected))
