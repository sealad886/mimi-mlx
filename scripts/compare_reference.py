from __future__ import annotations

import argparse

from mimi_mlx.cli import main as cli_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare MLX Mimi tokens with a reference backend")
    parser.add_argument("input")
    parser.add_argument("--weights", required=True)
    parser.add_argument(
        "--reference",
        choices=["rustymimi", "transformers"],
        default="transformers",
    )
    parser.add_argument("--reference-weights")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    argv = [
        "parity",
        args.input,
        "--reference",
        args.reference,
        "--weights",
        args.weights,
    ]
    if args.reference_weights:
        argv.extend(["--reference-weights", args.reference_weights])
    if args.json:
        argv.append("--json")
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
