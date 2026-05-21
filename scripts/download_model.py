#!/usr/bin/env python3
import sys, os, json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
OUTPUT_DIR = Path("/tmp/model_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"=== DOWNLOAD SCRIPT ===")
print(f"Model: {model_id}")

# Download using from_pretrained (no HF cache, direct to local dir)
print("Downloading model files...")
model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=False)
print("Model loaded.")

# Save to output dir
model.save_pretrained(save_directory=str(OUTPUT_DIR / model_id.replace("/", "_")))
print("Model saved.")

# Save tokenizer too
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(save_directory=str(OUTPUT_DIR / model_id.replace("/", "_")))
print("Tokenizer saved.")

# Create manifest
output_model = OUTPUT_DIR / model_id.replace("/", "_")
files = list(output_model.rglob("*"))
manifest = {"model_id": model_id, "files": []}
for f in files:
    if f.is_file():
        manifest["files"].append({"name": str(f.relative_to(output_model)), "size": f.stat().st_size})
        print(f"  {f.name} ({f.stat().st_size} bytes})")

with open(OUTPUT_DIR / "manifest.json", "w") as mf:
    json.dump(manifest, mf, indent=2)

total = sum(f["size"] for f in manifest["files"])
print(f"\nFiles: {len(manifest['files'])}, Total: {total/1024/1024:.1f} MB")
print("Done!")
