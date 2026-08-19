#!/usr/bin/env python3
"""Build a self-contained, checksummed SDK model bundle from expts8."""
import argparse, hashlib, json, shutil
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage1",required=True);p.add_argument("--contract",required=True)
    p.add_argument("--stage2",required=True);p.add_argument("--keywords",required=True)
    p.add_argument("--output",required=True);p.add_argument("--stage2-threshold",type=float,default=.7)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    copies={"stage1.onnx":a.stage1,"stage1_contract.json":a.contract,"stage2.onnx":a.stage2,"keywords.json":a.keywords}
    for name,source in copies.items(): shutil.copy2(source,out/name)
    manifest={"bundle_schema_version":1,"sdk_abi_version":1,"sdk_version":"0.0.2","sample_format":"pcm_s16le","sample_rate":16000,"channels":1,"stage1":"stage1.onnx","stage2":"stage2.onnx"}
    defaults={"stage2_threshold":a.stage2_threshold,"proposal_floor":-3.0,"competitor_beam":16,"token_prune":8,"pre_margin_frames":3,"post_margin_frames":0,"max_search_frames":128,"debounce_ms":1000,"max_segment_ms":180000}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (out/"sdk_defaults.json").write_text(json.dumps(defaults,indent=2)+"\n")
    names=sorted(set(copies)|{"manifest.json","sdk_defaults.json"})
    (out/"SHA256SUMS").write_text("".join(f"{digest(out/name)}  {name}\n" for name in names))
    print(out)
if __name__=="__main__":main()
