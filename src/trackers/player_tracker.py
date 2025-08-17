import os
import torch
import logging
from ultralytics import YOLO
import supervision as sv
from utils.stubs_utils import read_stub, save_stub

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlayerTracker:
    """
    Detect and track 'Player' objects across video frames using YOLO + ByteTrack.
    """

    def __init__(self, model_path, batch_size=20, device='cpu'):
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

    def detect_frames(self, frames, conf=0.5):
        """
        Batched detection for efficiency.
        """
        logger.info(f"Detecting players in {len(frames)} frames")
        
        detections = []
        for i in range(0, len(frames), self.batch_size):
            batch_frames = frames[i:i + self.batch_size]
            detections_batch = self.model(batch_frames, conf=conf, verbose=False)
            detections.extend(detections_batch)
            
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
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
            # Map YOLO class names to IDs
            cls_names_inv = {v: k for k, v in detection.names.items()}

            # Convert to Supervision format
            detection_sv = sv.Detections.from_ultralytics(detection)

            # Filter for 'Player' class before tracking
            if 'Player' in cls_names_inv:
                player_class_id = cls_names_inv['Player']
                detection_sv = detection_sv[detection_sv.class_id == player_class_id]

            tracked = self.tracker.update_with_detections(detection_sv)

            frame_tracks = {}
            if tracked.tracker_id is not None:
                for i, track_id in enumerate(tracked.tracker_id):
                    bbox = tracked.xyxy[i].tolist()
                    frame_tracks[int(track_id)] = {"bbox": bbox}
            tracks.append(frame_tracks)

        if stub_path:
            save_stub(stub_path, tracks)

        logger.info("Player tracking completed")
        return tracks
