import os
import logging
import numpy as np
import pandas as pd
from ultralytics import YOLO
import supervision as sv
from typing import List, Dict, Any
from utils.stubs_utils import read_stub, save_stub

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BallTracker:
    """
    A class that handles basketball detection and tracking using YOLO.

    This class provides methods to detect the ball in video frames, process detections
    in batches, and refine tracking results through filtering and interpolation.
    """

    def __init__(
        self, model_path: str, batch_size: int = 20, device: str = "cpu"
    ) -> None:
        """
        Args:
            model_path (str): Path to YOLO weights.
            batch_size (int): Frames per batch for inference.
            device (str): Device to run inference on ('cpu' or 'cuda').
        """
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = YOLO(model_path)
        self.model.to(device)
        self.tracker = sv.ByteTrack()
        self.batch_size = batch_size
        self.device = device

    def detect_frames(self, frames: List[Any], conf: float = 0.5) -> List[Any]:
        """
        Batched detection for efficiency.

        Args:
            frames (List[Any]): List of video frames (NumPy arrays or images).
            conf (float): Confidence threshold for detections.

        Returns:
            List[Any]: List of detection results for each frame.
        """
        detections = []
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i : i + self.batch_size]
            detections.extend(self.model(batch, conf=conf))
        return detections

    def get_object_tracks(
        self, frames: List[Any], read_from_stub: bool = False, stub_path: str = None
    ) -> List[Dict]:
        """
        Run detection + tracking with optional caching.

        Returns:
            list[dict]: One dict per frame mapping track_id -> bbox.
        """
        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None and len(tracks) == len(frames):
            logger.info(f"Loaded tracks from cache: {stub_path}")
            return tracks

        if len(frames) == 0:
            return [{}] * len(frames)

        # Determine ball class ID once
        sample_detection = self.model(frames[0:1], verbose=False)
        ball_class_id = next(
            (k for k, v in sample_detection[0].names.items() if v == "Ball"), None
        )
        if ball_class_id is None:
            logger.warning("No 'Ball' class found in model names")
            return [{}] * len(frames)

        detections = self.detect_frames(frames)
        tracks = []

        for detection in detections:
            detection_sv = sv.Detections.from_ultralytics(detection)
            tracks.append({})

            if len(detection_sv) == 0 or ball_class_id not in detection_sv.class_id:
                continue

            ball_mask = detection_sv.class_id == ball_class_id
            if ball_mask.any():
                ball_confs = detection_sv.confidence[ball_mask]
                max_idx = np.argmax(ball_confs)
                chosen_bbox = detection_sv.xyxy[ball_mask][max_idx].tolist()
                tracks[-1][1] = {"bbox": chosen_bbox}

        if stub_path is not None:
            save_stub(stub_path, tracks)
        return tracks

    def remove_wrong_detections(
        self, ball_positions: List[Dict[int, Dict[str, Any]]]
    ) -> List[Dict[int, Dict[str, Any]]]:
        """
        Remove wrong ball detections based on position criteria (temporal consistency check).

        Args:
            ball_positions (List[Dict[int, Dict[str, Any]]]): List of dicts mapping track_id -> ball data (must include "bbox").

        Returns:
            List[Dict[int, Dict[str, Any]]]: Filtered list of ball positions.
        """
        logger.info("Removing wrong ball detections...")
        maximum_allowed_distance = 25  # Pixels per frame; tunable for basketball speed
        last_good_frame_index = -1

        for i in range(len(ball_positions)):
            current_box = ball_positions[i].get(1, {}).get("bbox", [])

            if len(current_box) != 4:  # Ensure valid bbox
                continue

            if last_good_frame_index == -1:
                # First valid detection
                last_good_frame_index = i
                continue

            last_good_box = (
                ball_positions[last_good_frame_index].get(1, {}).get("bbox", [])
            )
            if len(last_good_box) != 4:
                continue

            frame_gap = i - last_good_frame_index
            adjusted_max_distance = maximum_allowed_distance * frame_gap

            # Compare top-left corners for position jump
            distance = np.linalg.norm(
                np.array(last_good_box[:2]) - np.array(current_box[:2])
            )

            if distance > adjusted_max_distance:
                ball_positions[i] = {}  # Discard invalid detection
            else:
                last_good_frame_index = i

        logger.info("Wrong ball detections removed.")
        return ball_positions

    def interpolate_missing_detections(
        self, ball_positions: List[Dict[int, Dict[str, Any]]]
    ) -> List[Dict[int, Dict[str, Any]]]:
        """
        Interpolate missing ball detections to create a smoother trajectory.

        Args:
            ball_positions (List[Dict[int, Dict[str, Any]]]): List of dicts mapping track_id -> ball data (must include "bbox").

        Returns:
            List[Dict[int, Dict[str, Any]]]: List of ball positions with interpolated values.
        """
        logger.info("Interpolating missing ball detections...")
        # Extract bboxes as list of [x1, y1, x2, y2] or []
        raw_bboxes = [pos.get(1, {}).get("bbox", []) for pos in ball_positions]

        # Create DataFrame; empty lists become NaN rows
        df = pd.DataFrame(raw_bboxes, columns=["x1", "y1", "x2", "y2"])

        # Forward fill any leading NaNs, then linear interpolate, then backward fill trailing
        df = df.ffill().interpolate(method="linear").bfill()

        # Reconstruct positions
        interpolated_positions = []
        for row in df.to_numpy():
            if np.isnan(row).any():
                interpolated_positions.append({})  # Still invalid if all NaN
            else:
                interpolated_positions.append({1: {"bbox": row.tolist()}})

        logger.info("Missing ball detections interpolated.")
        return interpolated_positions
