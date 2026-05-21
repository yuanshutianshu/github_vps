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
    local_path = snapshot_download(
        repo_id=model_id,
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"Return path: {local_path}")
    print(f"local_dir set to: {MODEL_DIR}")
    
    # Debug: what's in OUTPUT_DIR?
    print(f"\n=== OUTPUT_DIR contents ({OUTPUT_DIR}) ===")
    for item in sorted(OUTPUT_DIR.rglob("*")):
        rel = item.relative_to(OUTPUT_DIR)
        if item.is_dir():
            print(f"  [DIR]  {rel}")
        else:
            print(f"  [FILE] {rel} ({item.stat().st_size} bytes)")
    
    # Debug: what's in the returned local_path?
    lp = Path(local_path)
    print(f"\n=== Return path contents ({local_path}) ===")
    for item in sorted(lp.rglob("*")):
        rel = item.relative_to(lp)
        if item.is_dir():
            print(f"  [DIR]  {rel}")
        else:
            print(f"  [FILE] {rel} ({item.stat().st_size} bytes)")
    
    # Now use the correct path for the artifact
    # If local_path != MODEL_DIR, the actual model files are in local_path's parent
    actual_model_dir = lp if lp != MODEL_DIR else lp
    print(f"\n=== Actual model dir: {actual_model_dir} ===")
    
    manifest = {"model_id": model_id, "files": []}
    total_size = 0
    
    for f in sorted(actual_model_dir.rglob("*")):
        if f.is_file():
            rel = str(f.relative_to(actual_model_dir))
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
