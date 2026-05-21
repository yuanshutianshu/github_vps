#!/usr/bin/env python3
import sys, os, json, traceback
from pathlib import Path

model_id = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
OUTPUT_DIR = Path("/tmp/model_output")
MODEL_DIR = OUTPUT_DIR / model_id.replace("/", "_")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"=== DOWNLOAD SCRIPT ===")
print(f"Model: {model_id}")
print(f"Output: {MODEL_DIR}")

try:
    from huggingface_hub import snapshot_download
    
    print("Downloading...")
    # local_dir puts files directly there, no blobs/ in the path
    local_path = snapshot_download(
        repo_id=model_id,
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"Downloaded to: {local_path}")
    
    # List what's there
    local_p = Path(local_path)
    files = sorted(local_p.rglob("*"))
    print(f"Total items: {len(files)}")
    
    manifest = {"model_id": model_id, "files": []}
    total_size = 0
    for f in files:
        if f.is_file():
            rel = str(f.relative_to(local_p))
            sz = f.stat().st_size
            manifest["files"].append({"name": rel, "size": sz})
            total_size += sz
            print(f"  {rel} ({sz} bytes)")
    
    with open(OUTPUT_DIR / "manifest.json", "w") as mf:
        json.dump(manifest, mf, indent=2)
    
    print(f"\nFiles: {len(manifest['files'])}, Total: {total_size/1024/1024:.1f} MB")
    print("Done!")
    
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
