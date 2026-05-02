#!/usr/bin/env python3
"""Upload trained artifacts to ModelScope.

Set MODELSCOPE_TOKEN in the environment before running this script.
Large model files are intentionally excluded from the GitHub repository.
"""

import os

from modelscope.hub.api import HubApi


USER = os.environ.get("MODELSCOPE_USER", "oooooo0o")
REPO_NAME = os.environ.get(
    "MODELSCOPE_REPO_NAME",
    "qwen3.5-9b-philosophy-Hegel-Lacan-Zizek",
)
REPO_ID = f"{USER}/{REPO_NAME}"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_DIR = os.environ.get(
    "LORA_DIR",
    os.path.join(BASE_DIR, "lora_qwen9b_sep_philosophy_cot_3epochs"),
)
MERGED_DIR = os.environ.get(
    "MERGED_DIR",
    os.path.join(BASE_DIR, "lora_qwen9b_sep_philosophy_cot_3epochs_GGUF"),
)


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    token = require_env("MODELSCOPE_TOKEN")

    api = HubApi()
    api.login(access_token=token)

    print(f"Creating repo if needed: {REPO_ID}")
    api.create_repo(REPO_ID, private=True)

    if os.path.isdir(LORA_DIR):
        print(f"Uploading LoRA adapter from {LORA_DIR} to lora/ ...")
        api.upload_folder(
            folder_path=LORA_DIR,
            path_in_repo="lora",
            repo_id=REPO_ID,
        )
    else:
        print(f"Skipping LoRA upload; directory does not exist: {LORA_DIR}")

    if os.path.isdir(MERGED_DIR):
        print(f"Uploading merged weights from {MERGED_DIR} to merged/ ...")
        api.upload_folder(
            folder_path=MERGED_DIR,
            path_in_repo="merged",
            repo_id=REPO_ID,
        )
    else:
        print(f"Skipping merged upload; directory does not exist: {MERGED_DIR}")

    print(f"Done: https://modelscope.cn/models/{REPO_ID}")


if __name__ == "__main__":
    main()
