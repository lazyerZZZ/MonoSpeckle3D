from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PipelineConfig
from .data import discover_raw_triplets, group_split, tile_triplets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper-aligned pseudo-overlapped morphology reconstruction")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="split acquisition groups and crop aligned image triplets")
    prepare.add_argument("--input", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--tile-size", type=int, default=256)
    prepare.add_argument("--stride", type=int, default=256)
    prepare.add_argument("--seed", type=int, default=42)
    separate = commands.add_parser("separate", help="separate one mixed image with DivideNet V3")
    separate.add_argument("--input", required=True, type=Path)
    separate.add_argument("--checkpoint", required=True, type=Path)
    separate.add_argument("--output", required=True, type=Path)
    separate.add_argument("--tile-size", type=int, default=256)
    separate.add_argument("--device", choices=("cpu", "cuda", "mps"))
    reconstruct = commands.add_parser("reconstruct", help="run SIFT, interpolation and calibrated triangulation")
    reconstruct.add_argument("--left", required=True, type=Path, help="separated clear image")
    reconstruct.add_argument("--right", required=True, type=Path, help="separated blurred image")
    reconstruct.add_argument("--config", required=True, type=Path)
    reconstruct.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        triplets = discover_raw_triplets(args.input)
        splits = group_split((item.group_id for item in triplets), seed=args.seed)
        by_group = {item.group_id: item for item in triplets}
        manifest: dict[str, list[str]] = {}
        for split_name, group_ids in splits.items():
            split_dir = args.output / split_name
            tile_triplets((by_group[group_id] for group_id in group_ids), split_dir, args.tile_size, args.stride)
            manifest[split_name] = group_ids
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return 0
    if args.command == "separate":
        import cv2

        from .separation import load_dividenet_v3, save_grayscale, separate_image

        image = cv2.imread(str(args.input), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(args.input)
        model, device = load_dividenet_v3(args.checkpoint, args.device)
        clear, blurred = separate_image(image, model, device, args.tile_size)
        save_grayscale(clear, args.output / "clear.png")
        save_grayscale(blurred, args.output / "blurred.png")
        return 0
    from .pipeline import reconstruct_from_separated

    config = PipelineConfig.load(args.config)
    path = reconstruct_from_separated(args.left, args.right, config, args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
