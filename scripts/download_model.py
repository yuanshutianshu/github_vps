#!/usr/bin/env python3
import sys, os, json, shutil
from pathlib import Path

model_id = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
OUTPUT_DIR = Path("/tmp/model_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"=== DOWNLOAD SCRIPT ===")
print(f"Model: {model_id}")

from huggingface_hub import HfApi, snapshot_download

# Download using snapshot_download to output dir directly
# This stores files in ~/.cache/huggingface/ but returns the snapshot path
print("Downloading...")
local_path = snapshot_download(
    repo_id=model_id,
    cache_dir=str(OUTPUT_DIR / "hf_cache"),
    resume_download=True,
)
print(f"Snapshot: {local_path}")

# The snapshot path is the blobs directory - we need the actual repo files
# snapshot_download returns the "downloaded" path which is the repo snapshot dir
# List what's there
local_p = Path(local_path)
print(f"\nSnapshot contents:")
for f in sorted(local_p.rglob("*"))[:10]:
    print(f"  {f.relative_to(local_p)}")

# Copy to output dir
output_model = OUTPUT_DIR / model_id.replace("/", "_")
output_model.mkdir(parents=True, exist_ok=True)

manifest = {"model_id": model_id, "files": []}
total_size = 0

# The snapshot_download returns the "blobs" parent or the actual model dir?
# Actually it returns the path to the downloaded snapshot which is the repo_files dir
# We need to list the files properly
for f in sorted(local_p.rglob("*")):
    if f.is_file():
        rel = f.relative_to(local_p)
        dest = output_model / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        sz = f.stat().st_size
        manifest["files"].append({"name": str(rel), "size": sz})
        total_size += sz
        print(f"  {rel} ({sz} bytes})")

with open(OUTPUT_DIR / "manifest.json", "w") as mf:
    json.dump(manifest, mf, indent=2)

print(f"\nFiles: {len(manifest['files'])}, Total: {total_size/1024/1024:.1f} MB")
print("Done!")
