#!/usr/bin/env python3
import sys, os, json, shutil
from pathlib import Path

model_id = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
HF_CACHE = Path("/tmp/hf_cache")
OUTPUT_DIR = Path("/tmp/model_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from huggingface_hub import snapshot_download

print(f"Downloading {model_id}...")
local_path = snapshot_download(
    repo_id=model_id,
    cache_dir=str(HF_CACHE),
    resume_download=True,
)
print(f"Snapshot: {local_path}")

snap_path = Path(local_path)
output_model = OUTPUT_DIR / model_id.replace("/", "_")
output_model.mkdir(parents=True, exist_ok=True)

manifest = {"model_id": model_id, "files": []}
total_size = 0

for f in snap_path.rglob("*"):
    if f.is_file():
        rel = f.relative_to(snap_path)
        dest = output_model / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        sz = f.stat().st_size
        manifest["files"].append({"name": str(rel), "size": sz})
        total_size += sz
        print(f"  {rel} ({sz} bytes)")

with open(OUTPUT_DIR / "manifest.json", "w") as mf:
    json.dump(manifest, mf, indent=2)

print(f"\nFiles: {len(manifest['files'])}, Total: {total_size/1024/1024:.1f} MB")
print("Done!")
