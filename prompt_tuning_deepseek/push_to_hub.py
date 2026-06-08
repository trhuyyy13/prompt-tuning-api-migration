"""Push a soft-prompt checkpoint directory to the Hugging Face Hub.

Uploads everything under --checkpoint_dir as-is (soft_prompt.pt, tokenizer
files, training_state.pt, prompt_config.json, training_args.json, ...) so the
checkpoint can later be pulled back with `huggingface_hub.snapshot_download`
and pointed to as `--checkpoint_dir` for evaluate.py / test_forget_quality.py.
"""
import argparse
import os

from huggingface_hub import HfApi, create_repo


def parse_args():
    p = argparse.ArgumentParser(description="Push a checkpoint dir to the Hugging Face Hub")
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--repo_id", required=True,
                   help="e.g. <username>/depapi-soft-prompt-deepseek")
    p.add_argument("--token", default=None,
                   help="HF token with write access; defaults to $HF_TOKEN or the cached CLI login")
    p.add_argument("--private", action="store_true")
    p.add_argument("--commit_message", default="Upload soft prompt checkpoint")
    return p.parse_args()


def main():
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN")
    if token is None:
        raise SystemExit("No HF token found: pass --token or set the HF_TOKEN environment variable")

    api = HfApi(token=token)
    repo_url = create_repo(args.repo_id, token=token, private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        folder_path=args.checkpoint_dir,
        commit_message=args.commit_message,
    )
    print(f"[done] pushed {args.checkpoint_dir} -> {repo_url}")


if __name__ == "__main__":
    main()
