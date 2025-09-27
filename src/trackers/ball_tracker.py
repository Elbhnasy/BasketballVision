import os
import logging
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
            batch = frames[i:i + self.batch_size]
            detections.extend(self.model(batch, conf=conf))
        return detections
    
    def get_object_tracks(
        self, frames: list, read_from_stub: bool = False, stub_path: str = None
    ) -> list[dict]:
        """
        Run detection + tracking with optional caching.

        Returns:
            list[dict]: One dict per frame mapping track_id -> bbox.
        """

        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None and len(tracks) == len(frames):
            logger.info(f"Loaded tracks from cache: {stub_path}")
            return tracks

        logger.info("Running player detection and tracking...")
        detections = self.detect_frames(frames)
        tracks = []

        for frame_num, detection in enumerate(detections):
            # Convert to Supervision format
            detection_sv = sv.Detections.from_ultralytics(detection)

            tracks.append({}) 
            chosen_bbox = None
            max_conf = 0.0
            for frame_detection in detection_sv:
                bbox = frame_detection[0].tolist()
                confidence = frame_detection[2]
                cls_id = int(frame_detection[3])

                # Check if this detection is a ball using the class ID
                if cls_id in detection.names and detection.names[cls_id] == "Ball":
                    if confidence > max_conf:
                        max_conf = confidence
                        chosen_bbox = bbox

            if chosen_bbox is not None:
                tracks[frame_num][1] = {"bbox": chosen_bbox}
        if stub_path is not None:
            save_stub(stub_path, tracks)
        return tracks
            