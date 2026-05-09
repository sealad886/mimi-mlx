from __future__ import annotations

import argparse

from mimi_mlx.cli import main as cli_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark MLX Mimi encode")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    argv = ["benchmark", "encode", "--weights", args.weights, "--input-dir", args.input_dir]
    if args.json:
        argv.append("--json")
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
