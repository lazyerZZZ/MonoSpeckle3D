from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from PIL import Image


@dataclass(frozen=True)
class ImageTriplet:
    group_id: str
    blended: Path
    clear: Path
    blurred: Path


def discover_raw_triplets(input_dir: str | Path) -> list[ImageTriplet]:
    """Discover ordered BMP triplets using the acquisition order used in the thesis."""
    files = sorted(Path(input_dir).glob("*.bmp"))
    if not files:
        raise FileNotFoundError(f"No BMP images found in {input_dir}")
    if len(files) % 3:
        raise ValueError(f"Expected complete image triplets, found {len(files)} BMP files")
    return [
        ImageTriplet(str(index // 3 + 1), *files[index : index + 3])
        for index in range(0, len(files), 3)
    ]


def tile_positions(width: int, height: int, tile_size: int, stride: int) -> Iterator[tuple[int, int, int]]:
    tile_index = 1
    for top in range(0, height - tile_size + 1, stride):
        for left in range(0, width - tile_size + 1, stride):
            yield tile_index, left, top
            tile_index += 1


def tile_triplets(
    triplets: Iterable[ImageTriplet], output_dir: str | Path, tile_size: int = 256, stride: int = 256
) -> int:
    """Crop aligned triplets and retain the acquisition group in every filename."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sample_count = 0
    for triplet in triplets:
        paths = (triplet.blended, triplet.clear, triplet.blurred)
        images = [Image.open(path).convert("L") for path in paths]
        try:
            sizes = {image.size for image in images}
            if len(sizes) != 1:
                raise ValueError(f"Group {triplet.group_id} images have different sizes: {sizes}")
            width, height = images[0].size
            for tile_index, left, top in tile_positions(width, height, tile_size, stride):
                box = (left, top, left + tile_size, top + tile_size)
                for image, label in zip(images, ("blended", "clear", "blurred")):
                    image.crop(box).save(output / f"{triplet.group_id}_{tile_index}_{label}.png")
                sample_count += 1
        finally:
            for image in images:
                image.close()
    return sample_count


def group_split(group_ids: Iterable[str], train: float = 0.7, validation: float = 0.2, seed: int = 42) -> dict[str, list[str]]:
    """Split before cropping so neighboring patches cannot leak across subsets."""
    if train <= 0 or validation < 0 or train + validation >= 1:
        raise ValueError("Ratios must satisfy train > 0, validation >= 0 and train + validation < 1")
    groups = sorted(set(group_ids))
    random.Random(seed).shuffle(groups)
    train_end = round(len(groups) * train)
    validation_end = train_end + round(len(groups) * validation)
    return {
        "train": groups[:train_end],
        "validation": groups[train_end:validation_end],
        "test": groups[validation_end:],
    }
