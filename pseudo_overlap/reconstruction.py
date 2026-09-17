from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import median_filter

from .config import CameraCalibration, FilterConfig


def filter_displacement(u_field: np.ndarray, v_field: np.ndarray, config: FilterConfig) -> tuple[np.ndarray, np.ndarray]:
    if u_field.shape != v_field.shape:
        raise ValueError("u and v fields must have the same shape")
    invalid = ~np.isfinite(u_field) | ~np.isfinite(v_field)
    u_filled = np.nan_to_num(u_field, nan=0.0).astype(np.float32)
    v_filled = np.nan_to_num(v_field, nan=0.0).astype(np.float32)
    u_median = median_filter(u_filled, size=config.median_window)
    v_median = median_filter(v_filled, size=config.median_window)
    spikes = (np.abs(u_filled - u_median) > config.jump_threshold) | (np.abs(v_filled - v_median) > config.jump_threshold)
    u_smooth = cv2.bilateralFilter(u_filled, config.bilateral_diameter, config.bilateral_sigma_color, config.bilateral_sigma_space)
    v_smooth = cv2.bilateralFilter(v_filled, config.bilateral_diameter, config.bilateral_sigma_color, config.bilateral_sigma_space)
    invalid |= spikes
    u_smooth[invalid] = np.nan
    v_smooth[invalid] = np.nan
    return u_smooth, v_smooth


def triangulate(
    u_field: np.ndarray, v_field: np.ndarray, calibration: CameraCalibration, config: FilterConfig
) -> np.ndarray:
    height, width = u_field.shape
    y, x = np.mgrid[0:height, 0:width]
    margin = config.roi_margin
    roi = np.zeros((height, width), dtype=bool)
    if margin * 2 >= min(height, width):
        raise ValueError("roi_margin leaves no valid image region")
    roi[margin : height - margin, margin : width - margin] = True
    valid = roi & np.isfinite(u_field) & np.isfinite(v_field)
    left = np.column_stack((x[valid], y[valid])).astype(np.float32).reshape(-1, 1, 2)
    right = np.column_stack((x[valid] + u_field[valid], y[valid] + v_field[valid])).astype(np.float32).reshape(-1, 1, 2)
    if not len(left):
        return np.empty((0, 3), dtype=np.float64)
    left_ud = cv2.undistortPoints(left, calibration.left_intrinsic, calibration.left_distortion, P=calibration.left_intrinsic)
    right_ud = cv2.undistortPoints(right, calibration.right_intrinsic, calibration.right_distortion, P=calibration.right_intrinsic)
    p1 = calibration.left_intrinsic @ np.hstack((np.eye(3), np.zeros((3, 1))))
    p2 = calibration.right_intrinsic @ np.hstack((calibration.rotation, calibration.translation))
    homogeneous = cv2.triangulatePoints(p1, p2, left_ud.reshape(-1, 2).T, right_ud.reshape(-1, 2).T)
    points = (homogeneous[:3] / homogeneous[3]).T
    points = points[np.all(np.isfinite(points), axis=1)]
    if config.use_statistical_outlier_removal and len(points) >= config.sor_neighbors:
        try:
            import open3d as o3d
        except ImportError as error:
            raise RuntimeError("Open3D is required when statistical outlier removal is enabled") from error
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        _, indices = cloud.remove_statistical_outlier(config.sor_neighbors, config.sor_std_ratio)
        points = points[np.asarray(indices, dtype=int)]
    depth = points[:, 2]
    return points[(depth > config.min_depth) & (depth < config.max_depth)]
