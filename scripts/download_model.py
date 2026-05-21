#!/usr/bin/env python3
import sys, os, subprocess

model_id = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
target = sys.argv[2] if len(sys.argv) > 2 else ""

print(f"=== DOWNLOAD SCRIPT ===")
print(f"Model: {model_id}")
print(f"Target: {target}")
print(f"HF_TOKEN set: {bool(os.environ.get('HF_TOKEN'))}")

from huggingface_hub import snapshot_download
from pathlib import Path

cache_dir = Path("/tmp/model_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

print("Starting download...")
try:
    local_path = snapshot_download(
        repo_id=model_id,
        cache_dir=str(cache_dir),
        resume_download=True,
    )
    print(f"Downloaded to: {local_path}")
    
    # List files
    local_p = Path(local_path)
    files = list(local_p.rglob("*"))
    print(f"Total files: {len(files)}")
    for f in files[:10]:
        print(f"  {f.relative_to(local_p)}")
    
    # Write success marker
    with open("/tmp/download_success.txt", "w") as f:
        f.write(f"SUCCESS:{local_path}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
