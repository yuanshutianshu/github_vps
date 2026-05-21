#!/usr/bin/env python3
import sys, os, json
from pathlib import Path

model_id = sys.argv[1] if len(sys.argv) > 1 else "gpt2"

print(f"=== DOWNLOAD SCRIPT ===")
print(f"Model: {model_id}")

from huggingface_hub import snapshot_download

cache_dir = Path("/tmp/model_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

print("Downloading...")
local_path = snapshot_download(
    repo_id=model_id,
    cache_dir=str(cache_dir),
    resume_download=True,
)
print(f"Downloaded to: {local_path}")

# List files and write manifest
local_p = Path(local_path)
files = list(local_p.rglob("*"))
print(f"Total files: {len(files)}")

manifest = {"model_id": model_id, "local_path": str(local_path), "files": []}
for f in files:
    if f.is_file():
        size = f.stat().st_size
        manifest["files"].append({"name": str(f.relative_to(local_p)), "size": size})
        print(f"  {f.relative_to(local_p)} ({size} bytes)")

# Write manifest
with open("/tmp/model_manifest.json", "w") as mf:
    json.dump(manifest, mf, indent=2)
print(f"Manifest written with {len(manifest['files'])} files")

# Save the local path for next step
with open(os.environ["GITHUB_OUTPUT"], "a") as f:
    f.write(f"model_path={local_path}\n")
    f.write(f"file_count={len(manifest['files'])}\n")

print("Done!")
