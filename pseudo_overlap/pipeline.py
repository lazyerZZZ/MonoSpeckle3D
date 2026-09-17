from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import PipelineConfig
from .matching import sift_displacement
from .reconstruction import filter_displacement, triangulate


def reconstruct_from_separated(left_path: str | Path, right_path: str | Path, config: PipelineConfig, output_dir: str | Path) -> Path:
    left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
    if left is None:
        raise FileNotFoundError(left_path)
    if right is None:
        raise FileNotFoundError(right_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    matches = sift_displacement(left, right, config.matching)
    np.save(output / "raw_disp_u.npy", matches.dense_u)
    np.save(output / "raw_disp_v.npy", matches.dense_v)
    u_filtered, v_filtered = filter_displacement(matches.dense_u, matches.dense_v, config.filtering)
    np.save(output / "filtered_disp_u.npy", u_filtered)
    np.save(output / "filtered_disp_v.npy", v_filtered)
    points = triangulate(u_filtered, v_filtered, config.calibration, config.filtering)
    point_cloud_path = output / "point_cloud.xyz"
    np.savetxt(point_cloud_path, points, fmt="%.6f %.6f %.6f")
    return point_cloud_path
