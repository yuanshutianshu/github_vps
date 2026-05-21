#!/usr/bin/env python3
import sys, os, json, shutil
from pathlib import Path

model_id = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
HF_CACHE = Path("/tmp/hf_cache")
OUTPUT_DIR = Path("/tmp/model_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"=== DOWNLOAD SCRIPT ===")
print(f"Model: {model_id}")

from huggingface_hub import snapshot_download

print("Downloading...")
local_path = snapshot_download(
    repo_id=model_id,
    cache_dir=str(HF_CACHE),
    resume_download=True,
)
print(f"Downloaded to: {local_path}")

# List actual model files (not cache metadata)
local_p = Path(local_path)
all_files = list(local_p.rglob("*"))
print(f"Total items in cache: {len(all_files)}")

# Files to upload: everything EXCEPT .gitattributes, lock files, and HF cache metadata
skip_dirs = {'.git', '__pycache__', 'tmp', 'refs', '.cache'}
manifest_files = []

for f in all_files:
    if f.is_file():
        rel = str(f.relative_to(local_p))
        # Skip HF cache management files
        if any(s in rel.split(os.sep) for s in skip_dirs):
            continue
        # Skip lock files
        if f.suffix == '.lock':
            continue
        manifest_files.append(f)
        print(f"  {rel} ({f.stat().st_size} bytes)")

print(f"\nModel files to upload: {len(manifest_files)}")

# Copy to output dir
output_model = OUTPUT_DIR / model_id.replace("/", "_")
output_model.mkdir(parents=True, exist_ok=True)

manifest = {"model_id": model_id, "files": []}
for f in manifest_files:
    rel = f.relative_to(local_p)
    dest = output_model / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(f, dest)
    manifest["files"].append({"name": str(rel), "size": f.stat().st_size})

# Write manifest
with open(OUTPUT_DIR / "manifest.json", "w") as mf:
    json.dump(manifest, mf, indent=2)

print(f"\nOutput dir: {OUTPUT_DIR}")
print(f"Output files: {list(OUTPUT_DIR.rglob('*'))}")
