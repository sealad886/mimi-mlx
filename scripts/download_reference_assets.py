from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from mimi_mlx.config import DEFAULT_MIMI_REPO_ID, DEFAULT_MIMI_REVISION


def main() -> int:
    parser = argparse.ArgumentParser(description="Download official Mimi assets with the HF CLI")
    parser.add_argument("--repo-id", default=DEFAULT_MIMI_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_MIMI_REVISION)
    parser.add_argument("--output-dir", default="fixtures/reference/hf")
    parser.add_argument(
        "--enable-xet",
        action="store_true",
        help="Allow Hugging Face Xet transport. Disabled by default for resumable HTTP.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "hf",
        "download",
        args.repo_id,
        "--revision",
        args.revision,
        "--local-dir",
        str(output_dir),
        "--include",
        "config.json",
        "--include",
        "preprocessor_config.json",
        "--include",
        "model.safetensors",
    ]
    env = os.environ.copy()
    if not args.enable_xet:
        env["HF_HUB_DISABLE_XET"] = "1"
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
