#!/usr/bin/env python3
import os, sys
from huggingface_hub import snapshot_download, HfApi

model_id = sys.argv[1] if len(sys.argv) > 1 else ""
target_space = sys.argv[2] if len(sys.argv) > 2 else ""
token = os.environ.get("HF_TOKEN", "")

print("=== Downloading", model_id, "===")

dl_dir = snapshot_download(
    repo_id=model_id,
    token=token,
    cache_dir="/tmp/model_dl",
)

print("Downloaded to:", dl_dir)

for root, dirs, files in os.walk(dl_dir):
    for f in files:
        fp = os.path.join(root, f)
        size = os.path.getsize(fp)
        print(" ", fp, size // 1024, "KB")

if target_space:
    api = HfApi(token=token)
    me = api.whoami()
    username = me["name"]
    space_name = target_space.split("/")[-1] if "/" in target_space else target_space
    full_repo = username + "/" + space_name
    try:
        api.create_repo(
            repo_id=full_repo,
            repo_type="space",
            space_sdk="docker",
            space_hardware="small",
            exist_ok=True,
            repo_visibility="private",
        )
        print("Space", full_repo, "ready")
    except Exception as e:
        print("Space note:", e)
    print("Uploading...")
    api.upload_folder(
        folder_path=dl_dir,
        repo_id=full_repo,
        repo_type="model",
        commit_message="Upload " + model_id,
    )
    print("Uploaded! https://huggingface.co/" + full_repo)
else:
    print("No upload (target_space empty)")

print("DONE")
