from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="kyutai/mimi")
    parser.add_argument("--revision")
    parser.add_argument("--output-dir", default="fixtures/reference")
    parser.parse_args()
    raise SystemExit("Reference asset download is implemented in Stage 2/5")


if __name__ == "__main__":
    raise SystemExit(main())
