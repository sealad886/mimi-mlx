from __future__ import annotations

import argparse

from mimi_mlx.cli import main as cli_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare MLX Mimi tokens with Transformers")
    parser.add_argument("input")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    argv = [
        "parity",
        args.input,
        "--reference",
        "transformers",
        "--weights",
        args.weights,
    ]
    if args.json:
        argv.append("--json")
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
