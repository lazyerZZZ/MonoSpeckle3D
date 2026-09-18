from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PipelineConfig
from .data import discover_raw_triplets, group_split, tile_triplets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pseudo-overlapped morphology reconstruction")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="make the fixed 7:2:1, 256-pixel training dataset")
    prepare.add_argument("--input", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)

    train = commands.add_parser("train", help="train DivideNet V3 for 200 epochs")
    train.add_argument("--dataset", required=True, type=Path)
    train.add_argument("--checkpoints", required=True, type=Path)

    run = commands.add_parser("run", help="run the complete image-to-point-cloud pipeline")
    run.add_argument("--input", required=True, type=Path)
    run.add_argument("--checkpoint", required=True, type=Path)
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    return parser


def prepare_dataset(input_dir: Path, output_dir: Path) -> None:
    triplets = discover_raw_triplets(input_dir)
    splits = group_split((triplet.group_id for triplet in triplets), train=0.7, validation=0.2, seed=42)
    triplet_by_group = {triplet.group_id: triplet for triplet in triplets}
    for split_name, group_ids in splits.items():
        tile_triplets(
            (triplet_by_group[group_id] for group_id in group_ids),
            output_dir / split_name,
            tile_size=256,
            stride=256,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split_manifest.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        prepare_dataset(args.input, args.output)
        return 0
    if args.command == "train":
        from .training import train_dividenet

        print(train_dividenet(args.dataset, args.checkpoints))
        return 0

    from .pipeline import run_pipeline

    config = PipelineConfig.load(args.config)
    print(run_pipeline(args.input, args.checkpoint, config, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
