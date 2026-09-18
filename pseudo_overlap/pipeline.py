from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import PipelineConfig
from .matching import sift_displacement
from .reconstruction import filter_displacement, triangulate
from .separation import load_dividenet_v3, save_grayscale, separate_image


def run_pipeline(
    mixed_image_path: str | Path,
    checkpoint_path: str | Path,
    config: PipelineConfig,
    output_dir: str | Path,
) -> Path:
    """Run the thesis pipeline from one pseudo-overlapped image to an XYZ cloud."""
    mixed = cv2.imread(str(mixed_image_path), cv2.IMREAD_GRAYSCALE)
    if mixed is None:
        raise FileNotFoundError(mixed_image_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    model, device = load_dividenet_v3(checkpoint_path)
    clear, blurred = separate_image(mixed, model, device)
    clear_path = output / "clear.png"
    blurred_path = output / "blurred.png"
    save_grayscale(clear, clear_path)
    save_grayscale(blurred, blurred_path)

    clear_u8 = np.clip(clear * 255.0, 0, 255).astype(np.uint8)
    blurred_u8 = np.clip(blurred * 255.0, 0, 255).astype(np.uint8)
    matches = sift_displacement(clear_u8, blurred_u8, config.matching)
    np.save(output / "raw_disp_u.npy", matches.dense_u)
    np.save(output / "raw_disp_v.npy", matches.dense_v)
    u_filtered, v_filtered = filter_displacement(matches.dense_u, matches.dense_v, config.filtering)
    np.save(output / "filtered_disp_u.npy", u_filtered)
    np.save(output / "filtered_disp_v.npy", v_filtered)
    points = triangulate(u_filtered, v_filtered, config.calibration, config.filtering)
    point_cloud_path = output / "point_cloud.xyz"
    np.savetxt(point_cloud_path, points, fmt="%.6f %.6f %.6f")
    return point_cloud_path
