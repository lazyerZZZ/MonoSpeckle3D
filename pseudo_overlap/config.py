from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


@dataclass(frozen=True)
class CameraCalibration:
    left_intrinsic: np.ndarray
    right_intrinsic: np.ndarray
    left_distortion: np.ndarray
    right_distortion: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraCalibration":
        return cls(
            left_intrinsic=_array(data["left_intrinsic"], (3, 3), "left_intrinsic"),
            right_intrinsic=_array(data["right_intrinsic"], (3, 3), "right_intrinsic"),
            left_distortion=np.asarray(data["left_distortion"], dtype=np.float64).reshape(-1),
            right_distortion=np.asarray(data["right_distortion"], dtype=np.float64).reshape(-1),
            rotation=_array(data["rotation"], (3, 3), "rotation"),
            translation=np.asarray(data["translation"], dtype=np.float64).reshape(3, 1),
        )


@dataclass(frozen=True)
class MatchConfig:
    max_features: int = 20_000
    ratio_threshold: float = 0.7
    ransac_threshold: float = 3.0
    min_matches: int = 8
    interpolation: str = "linear"


@dataclass(frozen=True)
class FilterConfig:
    median_window: int = 5
    jump_threshold: float = 0.5
    bilateral_diameter: int = 9
    bilateral_sigma_color: float = 0.5
    bilateral_sigma_space: float = 5.0
    roi_margin: int = 200
    min_depth: float = 50.0
    max_depth: float = 500.0
    use_statistical_outlier_removal: bool = True
    sor_neighbors: int = 30
    sor_std_ratio: float = 2.5


@dataclass(frozen=True)
class PipelineConfig:
    calibration: CameraCalibration
    matching: MatchConfig = field(default_factory=MatchConfig)
    filtering: FilterConfig = field(default_factory=FilterConfig)

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(
            calibration=CameraCalibration.from_dict(data["calibration"]),
            matching=MatchConfig(**data.get("matching", {})),
            filtering=FilterConfig(**data.get("filtering", {})),
        )
