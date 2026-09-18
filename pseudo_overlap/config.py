from __future__ import annotations

import json
from dataclasses import dataclass
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
    max_features: int
    ratio_threshold: float
    ransac_threshold: float
    min_matches: int


@dataclass(frozen=True)
class FilterConfig:
    median_window: int
    jump_threshold: float
    bilateral_diameter: int
    bilateral_sigma_color: float
    bilateral_sigma_space: float
    roi_margin: int
    min_depth: float
    max_depth: float
    sor_neighbors: int
    sor_std_ratio: float


@dataclass(frozen=True)
class PipelineConfig:
    calibration: CameraCalibration
    matching: MatchConfig
    filtering: FilterConfig

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(
            calibration=CameraCalibration.from_dict(data["calibration"]),
            matching=MatchConfig(**data["matching"]),
            filtering=FilterConfig(**data["filtering"]),
        )
