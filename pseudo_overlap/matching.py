from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.interpolate import griddata

from .config import MatchConfig


@dataclass(frozen=True)
class MatchResult:
    left_points: np.ndarray
    right_points: np.ndarray
    dense_u: np.ndarray
    dense_v: np.ndarray


def sift_displacement(left: np.ndarray, right: np.ndarray, config: MatchConfig) -> MatchResult:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("SIFT inputs must be grayscale images with identical shapes")
    sift = cv2.SIFT_create(nfeatures=config.max_features)
    left_keypoints, left_descriptors = sift.detectAndCompute(left, None)
    right_keypoints, right_descriptors = sift.detectAndCompute(right, None)
    if left_descriptors is None or right_descriptors is None:
        raise RuntimeError("SIFT could not find descriptors in both images")
    pairs = cv2.BFMatcher().knnMatch(left_descriptors, right_descriptors, k=2)
    good = [first for first, second in pairs if first.distance < config.ratio_threshold * second.distance]
    if len(good) < config.min_matches:
        raise RuntimeError(f"Only {len(good)} ratio-test matches; need at least {config.min_matches}")
    left_points = np.float32([left_keypoints[item.queryIdx].pt for item in good])
    right_points = np.float32([right_keypoints[item.trainIdx].pt for item in good])
    _, mask = cv2.findFundamentalMat(left_points, right_points, cv2.FM_RANSAC, config.ransac_threshold, 0.99)
    if mask is None:
        raise RuntimeError("RANSAC could not estimate a fundamental matrix")
    inliers = mask.ravel().astype(bool)
    left_points, right_points = left_points[inliers], right_points[inliers]
    if len(left_points) < config.min_matches:
        raise RuntimeError(f"Only {len(left_points)} RANSAC inliers; need at least {config.min_matches}")
    displacement = right_points - left_points
    height, width = left.shape
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    dense_u = griddata(left_points, displacement[:, 0], (grid_x, grid_y), method="linear")
    dense_v = griddata(left_points, displacement[:, 1], (grid_x, grid_y), method="linear")
    return MatchResult(left_points, right_points, dense_u, dense_v)
