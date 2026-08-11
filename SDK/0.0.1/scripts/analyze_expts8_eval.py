#!/usr/bin/env python3
"""Merge SDK array shards and compute per-dataset FR/FA/confusion metrics."""
import argparse, json
from collections import Counter
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--shards-root",required=True);p.add_argument("--keywords",required=True);p.add_argument("--output",required=True);a=p.parse_args()
root=Path(a.shards_root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
keys=json.load(open(a.keywords))["keywords"];by_text={x["display_text"].casefold():x["id"] for x in keys}
summary={"datasets":{},"total_records":0,"total_errors":0}
for dataset in sorted(x for x in root.iterdir() if x.is_dir()):
    rows=[]
    for part in sorted(dataset.glob("part-*.jsonl")):
        rows.extend(json.loads(line) for line in part.open() if line.strip())
    rows.sort(key=lambda x:x.get("source_index",-1));label=int((dataset/"expected_label").read_text());errors=sum("error" in x for x in rows);detected=sum(bool(x.get("events")) for x in rows);event_count=sum(len(x.get("events",[])) for x in rows);audio_hours=sum(x.get("samples",0) for x in rows)/16000/3600;correct=0;wrong=0
    confusion=Counter()
    if label:
        for row in rows:
            expected=row.get("expected_keyword_id") or by_text.get(row.get("text","").casefold(),"")
            ids=[e["keyword_id"] for e in row.get("events",[])]
            if expected and expected in ids:correct+=1
            elif ids:wrong+=1;confusion[(expected,ids[0])]+=1
    rtfs=[x["rtf"] for x in rows if "rtf" in x]
    metrics={"expected_label":label,"records":len(rows),"errors":errors,"detected":detected,"event_count":event_count,"audio_hours":audio_hours,"events_per_hour":event_count/max(audio_hours,1e-12),"rtf_mean":sum(rtfs)/max(1,len(rtfs)),"rtf_max":max(rtfs,default=0)}
    if label:metrics.update(correct=correct,wrong_keyword=wrong,false_reject=len(rows)-errors-correct,false_reject_rate=(len(rows)-errors-correct)/max(1,len(rows)-errors),confusions=[{"expected":x[0],"detected":x[1],"count":n} for x,n in confusion.most_common()])
    else:metrics.update(false_accept=detected,false_accept_rate=detected/max(1,len(rows)-errors))
    summary["datasets"][dataset.name]=metrics;summary["total_records"]+=len(rows);summary["total_errors"]+=errors
    with (out/f"{dataset.name}.jsonl").open("w") as f:
        for row in rows:f.write(json.dumps(row,separators=(",",":"))+"\n")
(out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps({"records":summary["total_records"],"errors":summary["total_errors"],"datasets":len(summary["datasets"])}))
