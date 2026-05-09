from __future__ import annotations

import argparse
import json

from .parity import ParityStatus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mimi-mlx")
    subcommands = parser.add_subparsers(dest="command", required=True)

    encode = subcommands.add_parser("encode", help="Encode audio to Mimi tokens")
    encode.add_argument("input")
    encode.add_argument("--weights", required=True)
    encode.add_argument("--output", required=True)
    encode.add_argument("--json", action="store_true")

    decode = subcommands.add_parser("decode", help="Decode Mimi tokens to audio")
    decode.add_argument("tokens")
    decode.add_argument("--weights", required=True)
    decode.add_argument("--output", required=True)
    decode.add_argument("--json", action="store_true")

    parity = subcommands.add_parser("parity", help="Compare MLX tokens with a reference backend")
    parity.add_argument("input")
    parity.add_argument("--reference", choices=["rustymimi", "transformers"], required=True)
    parity.add_argument("--weights", required=True)
    parity.add_argument("--json", action="store_true")

    benchmark = subcommands.add_parser("benchmark", help="Run Mimi MLX benchmarks")
    benchmark_subcommands = benchmark.add_subparsers(dest="benchmark_command", required=True)
    for name in ("encode", "decode", "batching"):
        command = benchmark_subcommands.add_parser(name, help=f"Benchmark {name}")
        command.add_argument("--input-dir")
        command.add_argument("--batch-sizes", default="1")
        command.add_argument("--json", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "parity":
        status = ParityStatus.current()
        if getattr(args, "json", False):
            print(json.dumps(status.__dict__, sort_keys=True))
        else:
            print(status.reason)
        return 2

    parser.error(f"{args.command!r} is not implemented until model parity stages land")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
