import logging
from typing import List, Dict, Any
from .utils import draw_triangle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BallTracksDrawer:
    """
    Draws ball tracks on batches of video frames.

    Attributes:
        ball_pointer_color (Tuple[int, int, int]): BGR color for ball pointers.
    """

    DEFAULT_BALL_POINTER_COLOR: tuple[int, int, int] = (0, 255, 0)

    def __init__(
        self,
        ball_pointer_color: tuple[int, int, int] = DEFAULT_BALL_POINTER_COLOR,
    ) -> None:
        """
        Initialize the BallTracksDrawer.

        Args:
            ball_pointer_color: BGR color for ball pointers.
        """
        self.ball_pointer_color = ball_pointer_color

    def draw_batch(
        self,
        video_frames: List[Any],
        tracks_batch: List[Dict[int, Dict[str, Any]]],
    ) -> List[Any]:
        """
        Draw ball tracks on a batch of frames.

        Args:
            video_frames: List of frames (NumPy arrays or images).
            tracks_batch: List of dicts mapping track_id -> ball data (must include "bbox").

        Returns:
            List of frames with ball tracks drawn.
        """
        if not (len(video_frames) == len(tracks_batch)):
            raise ValueError("Input lists must have the same length.")

        output_frames = []
        append_frame = output_frames.append
        total_balls_drawn = 0

        for frame_idx, (frame, ball_dict) in enumerate(
            zip(video_frames, tracks_batch)
        ):
            frame_copy = frame.copy()
            balls_drawn = 0

            for _, ball in ball_dict.items():
                bbox = ball.get("bbox")
                if bbox is None:
                    continue

                frame_copy = draw_triangle(frame_copy, bbox, self.ball_pointer_color)
                balls_drawn += 1

            total_balls_drawn += balls_drawn
            append_frame(frame_copy)

            logger.debug(f"Processed frame {frame_idx}: drew {balls_drawn} ball pointers.")

        logger.info(f"Completed drawing on {len(output_frames)} frames, total balls drawn: {total_balls_drawn}.")
        return output_frames