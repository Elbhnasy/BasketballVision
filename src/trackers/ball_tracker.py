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
            
    def remove_wrong_detections(self, ball_positions: List[Dict[int, Dict[str, Any]]]) -> List[Dict[int, Dict[str, Any]]]:
        """
        Remove wrong ball detections based on position and size criteria.

        Args:
            ball_positions (List[Dict[int, Dict[str, Any]]]): List of dicts mapping track_id -> ball data (must include "bbox").

        Returns:
            List[Dict[int, Dict[str, Any]]]: Filtered list of ball positions.
        """
        maximum_allowed_distance = 25
        last_good_frame_index = -1

        for i in range(len(ball_positions)):
            current_box = ball_positions[i].get(1, {}).get('bbox', [])

            if len(current_box) == 0:
                continue

            if last_good_frame_index == -1:
                # First valid detection
                last_good_frame_index = i
                continue

            last_good_box = ball_positions[last_good_frame_index].get(1, {}).get('bbox', [])
            frame_gap = i - last_good_frame_index
            adjusted_max_distance = maximum_allowed_distance * frame_gap

            if np.linalg.norm(np.array(last_good_box[:2]) - np.array(current_box[:2])) > adjusted_max_distance:
                ball_positions[i] = {}
            else:
                last_good_frame_index = i

        return ball_positions

    def interpolate_missing_detections(self, ball_positions: List[Dict[int, Dict[str, Any]]]) -> List[Dict[int, Dict[str, Any]]]:
        """
        Interpolate missing ball detections to create a smoother trajectory.

        Args:
            ball_position (List[Dict[int, Dict[str, Any]]]): List of dicts mapping track_id -> ball data (must include "bbox").

        Returns:
            List[Dict[int, Dict[str, Any]]]: List of ball positions with interpolated values.
        """
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions