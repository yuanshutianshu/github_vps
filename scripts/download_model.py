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
    from huggingface_hub import HfApi, list_repo_files
    
    api = HfApi()
    
    # Get all files in the repo
    print("Getting file list...")
    files = list(list_repo_files(model_id))
    print(f"Total files: {len(files)}")
    for f in files:
        print(f"  {f}")
    
    # Download each file using hf_hub_download
    manifest = {"model_id": model_id, "files": []}
    total_size = 0
    
    for i, file in enumerate(files):
        dest = MODEL_DIR / file
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"[{i+1}/{len(files)}] Downloading {file}...")
        downloaded = api.hf_hub_download(
            repo_id=model_id,
            filename=file,
            local_dir=str(MODEL_DIR),
            local_dir_use_symlinks=False,
        )
        
        # Move from temp location to destination
        downloaded_p = Path(downloaded)
        if downloaded_p != dest:
            import shutil
            shutil.move(str(downloaded_p), str(dest))
        
        sz = dest.stat().st_size
        manifest["files"].append({"name": file, "size": sz})
        total_size += sz
        print(f"  -> {dest.name} ({sz} bytes)")
    
    with open(OUTPUT_DIR / "manifest.json", "w") as mf:
        json.dump(manifest, mf, indent=2)
    
    print(f"\nFiles: {len(manifest['files'])}, Total: {total_size/1024/1024:.1f} MB")
    print("Done!")
    
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
