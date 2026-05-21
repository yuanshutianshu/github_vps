#!/usr/bin/env python3
import sys, os, json, traceback
from pathlib import Path

model_id = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
OUTPUT_DIR = Path("/tmp/model_output")
MODEL_DIR = OUTPUT_DIR / model_id.replace("/", "_")

print(f"=== DOWNLOAD SCRIPT ===")
print(f"Model: {model_id}")
print(f"Output: {MODEL_DIR}")

try:
    import subprocess
    
    # git clone the model repo (no auth needed for public models)
    hf_url = f"https://huggingface.co/{model_id}"
    print(f"Cloning from {hf_url}...")
    
    result = subprocess.run(
        ["git", "clone", "--depth", "1", hf_url, str(MODEL_DIR)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    
    if result.returncode != 0:
        print(f"Clone failed: {result.stderr}")
        sys.exit(1)
    
    print("Clone done.")
    
    # List files
    files = sorted(MODEL_DIR.rglob("*"))
    manifest = {"model_id": model_id, "files": []}
    total_size = 0
    
    for f in files:
        if f.is_file():
            rel = str(f.relative_to(MODEL_DIR))
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
