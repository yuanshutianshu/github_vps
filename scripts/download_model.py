#!/usr/bin/env python3
import sys, os, json, shutil
from pathlib import Path

model_id = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
output_dir = Path("/tmp/model_output")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"=== DOWNLOAD SCRIPT ===")
print(f"Model: {model_id}")

from huggingface_hub import snapshot_download

print("Downloading to /tmp/model_output...")
local_path = snapshot_download(
    repo_id=model_id,
    cache_dir="/tmp/model_output",
    resume_download=True,
)
print(f"Downloaded to: {local_path}")

# Copy all files to output dir preserving structure
local_p = Path(local_path)
output_model = output_dir / model_id.replace("/", "_")
output_model.mkdir(parents=True, exist_ok=True)

files = list(local_p.rglob("*"))
print(f"Total files: {len(files)}")

manifest = {"model_id": model_id, "files": []}
for f in files:
    if f.is_file():
        rel = f.relative_to(local_p)
        dest = output_model / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        size = f.stat().st_size
        manifest["files"].append({"name": str(rel), "size": size})
        print(f"  {rel} ({size} bytes)")

# Write manifest
manifest_path = output_dir / "manifest.json"
with open(manifest_path, "w") as mf:
    json.dump(manifest, mf, indent=2)

print(f"\nManifest written: {manifest_path}")
print(f"Output dir contents: {list(output_dir.rglob('*'))}")

# Write GITHUB_OUTPUT
with open(os.environ["GITHUB_OUTPUT"], "a") as go:
    go.write(f"model_path={local_path}\n")
    go.write(f"output_dir={output_dir}\n")
    go.write(f"file_count={len(manifest['files'])}\n")

print("Done!")
