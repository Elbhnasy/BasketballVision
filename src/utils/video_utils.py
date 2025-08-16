import cv2
import os
import logging
import numpy as np
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_video(video_path: str) -> List[np.ndarray]:
    """
    Read all frames from a video file into memory.
    
    Args:
        video_path (str): Path to the input video file.
        
    Returns:
        List[np.ndarray]: List of video frames as numpy arrays.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)

    frames = []
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_count += 1
    finally:
        cap.release()
    
    logger.info(f"Read {len(frames)} frames from video: {video_path}")
    return frames

def save_video(output_video_frames: List[np.ndarray], output_video_path: str, 
                fps: float = 24.0, fourcc_code: str = 'mp4v') -> None:
    """
    Save a sequence of frames as a video file.
    Creates necessary directories if they don't exist and writes frames using specified codec.
    
    Args:
        output_video_frames (List[np.ndarray]): List of frames to save.
        output_video_path (str): Path where the video should be saved.
        fps (float): Frames per second for the output video. Defaults to 24.0.
        fourcc_code (str): FourCC code for video codec. Defaults to 'mp4v'.
    """
    if not output_video_frames:
        raise ValueError("No frames provided to save")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get frame dimensions from first frame
    first_frame = output_video_frames[0]
    height, width = first_frame.shape[:2]
    
    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    try:
        # Write all frames
        for frame in output_video_frames:
            out.write(frame)
    finally:
        out.release()
    
    logger.info(f"Saved video with {len(output_video_frames)} frames to {output_video_path}")
