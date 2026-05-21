#!/usr/bin/env python3
import sys, os, json, traceback
from pathlib import Path

model_id = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
OUTPUT_DIR = Path("/tmp/model_output")
MODEL_DIR = OUTPUT_DIR / model_id.replace("/", "_")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

print(f"=== DOWNLOAD SCRIPT ===")
print(f"Model: {model_id}")
print(f"Output: {MODEL_DIR}")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.save_pretrained(save_directory=str(MODEL_DIR))
    
    print("Downloading tokenizer...")
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.save_pretrained(save_directory=str(MODEL_DIR))
    
    # List output
    files = sorted(MODEL_DIR.rglob("*"))
    manifest = {"model_id": model_id, "files": []}
    total_size = 0
    for f in files:
        if f.is_file():
            sz = f.stat().st_size
            manifest["files"].append({"name": str(f.relative_to(MODEL_DIR)), "size": sz})
            total_size += sz
            print(f"  {f.name} ({sz} bytes)")
    
    with open(OUTPUT_DIR / "manifest.json", "w") as mf:
        json.dump(manifest, mf, indent=2)
    
    print(f"\nFiles: {len(manifest['files'])}, Total: {total_size/1024/1024:.1f} MB")
    print("Done!")
    
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
