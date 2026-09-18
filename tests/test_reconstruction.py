import unittest

import numpy as np

from pseudo_overlap.config import CameraCalibration, FilterConfig
from pseudo_overlap.reconstruction import triangulate


class ReconstructionTest(unittest.TestCase):
    def test_rectified_constant_disparity_recovers_depth(self):
        intrinsic = np.array([[100.0, 0.0, 3.5], [0.0, 100.0, 3.5], [0.0, 0.0, 1.0]])
        calibration = CameraCalibration(
            intrinsic,
            intrinsic.copy(),
            np.zeros(5),
            np.zeros(5),
            np.eye(3),
            np.array([[-1.0], [0.0], [0.0]]),
        )
        u = np.full((8, 8), -10.0, dtype=np.float32)
        v = np.zeros_like(u)
        config = FilterConfig(
            median_window=5,
            jump_threshold=0.5,
            bilateral_diameter=9,
            bilateral_sigma_color=0.5,
            bilateral_sigma_space=5.0,
            roi_margin=1,
            min_depth=1.0,
            max_depth=20.0,
            sor_neighbors=100,
            sor_std_ratio=2.5,
        )
        points = triangulate(u, v, calibration, config)
        self.assertEqual(len(points), 36)
        self.assertTrue(np.allclose(points[:, 2], 10.0, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
