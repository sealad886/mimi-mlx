from __future__ import annotations

import argparse

from mimi_mlx.cli import main as cli_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark MLX Mimi batch encode")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    argv = [
        "benchmark",
        "batching",
        "--weights",
        args.weights,
        "--input-dir",
        args.input_dir,
        "--batch-sizes",
        args.batch_sizes,
    ]
    if args.json:
        argv.append("--json")
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
