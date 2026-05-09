from __future__ import annotations

import argparse
import json

from mimi_mlx.weights import WeightLoadError, validate_hf_mimi_header


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a Mimi safetensors header")
    parser.add_argument("weights")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        manifest = validate_hf_mimi_header(args.weights)
    except WeightLoadError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        else:
            print(f"error: {exc}")
        return 2

    data = {
        "ok": True,
        "path": str(manifest.path),
        "tensor_count": manifest.tensor_count,
        "required_count": manifest.required_count,
        "total_parameters": manifest.total_parameters,
    }
    if args.json:
        print(json.dumps(data, sort_keys=True))
    else:
        print(
            f"{data['tensor_count']} tensors, {data['total_parameters']} parameters, "
            f"{data['required_count']} required tensors present"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
