"""
Homography Coordinate Transformer.

Loads the pre-computed homography matrix from calibration.yaml (produced by
the web-based calibration tool at /calibrate-ui) and maps 2D bounding box
bottom-centers to real-world ground-plane coordinates during the micro-loop.
"""

from typing import Dict, Tuple

import cv2
import numpy as np
import yaml  # type: ignore


class CoordinateTransformer:
    """
    Real-time execution class that loads the pre-computed homography matrix
    and transforms 2D bounding boxes into real-world coordinates (metres).
    """

    def __init__(self, calibration_file: str = "calibration.yaml") -> None:
        """
        Loads the homography matrix from the calibration YAML file.

        Args:
            calibration_file (str): Path to the calibration YAML produced
                by the web calibration tool. Defaults to 'calibration.yaml'.

        Raises:
            FileNotFoundError: If the calibration file does not exist.
                Run the web calibration tool at /calibrate-ui first.
        """
        try:
            with open(calibration_file, "r") as f:
                calib_data = yaml.safe_load(f)
            self.H = np.array(calib_data["homography"], dtype=np.float32)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Calibration file '{calibration_file}' not found. "
                "Run the web calibration tool at /calibrate-ui first."
            )

    def get_real_world_coords(
        self, tracked_boxes: np.ndarray
    ) -> Dict[int, Tuple[float, float]]:
        """
        Maps the bottom-center of each tracked bounding box to the real-world
        ground plane using the loaded homography matrix.

        Args:
            tracked_boxes (np.ndarray): Shape (N, >=5).
                Expected format per row: [x1, y1, x2, y2, track_id, ...]

        Returns:
            Dict[int, Tuple[float, float]]: Mapping of
                {track_id: (real_x, real_y)} in metres.
        """
        if len(tracked_boxes) == 0:
            return {}

        real_world_positions: Dict[int, Tuple[float, float]] = {}

        for box_data in tracked_boxes:
            x1, y1, x2, y2, track_id = box_data[:5]

            # Bottom-center is the point where the vehicle contacts the ground.
            bottom_center_x = (x1 + x2) / 2.0
            bottom_center_y = y2

            # perspectiveTransform requires shape (1, 1, 2)
            pts = np.array(
                [[[bottom_center_x, bottom_center_y]]], dtype=np.float32
            )
            world_pts = cv2.perspectiveTransform(pts, self.H)

            real_world_positions[int(track_id)] = (
                float(world_pts[0][0][0]),
                float(world_pts[0][0][1]),
            )

        return real_world_positions
