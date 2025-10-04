import logging
from typing import List, Dict, Any, Tuple
from .utils import draw_ellipse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlayerTracksDrawer:
    """
    Draws player tracks on batches of video frames.

    Attributes:
        default_player_team_id (int): Default team ID when not specified.
        team_1_color (Tuple[int, int, int]): RGB color for Team 1.
        team_2_color (Tuple[int, int, int]): RGB color for Team 2.
    """

    DEFAULT_TEAM_1_COLOR: Tuple[int, int, int] = (255, 245, 238)
    DEFAULT_TEAM_2_COLOR: Tuple[int, int, int] = (128, 128, 0)

    def __init__(
        self,
        team_1_color: Tuple[int, int, int] = DEFAULT_TEAM_1_COLOR,
        team_2_color: Tuple[int, int, int] = DEFAULT_TEAM_2_COLOR,
    ) -> None:
        """
        Initialize the PlayerTracksDrawer.

        Args:
            team_1_color: RGB color for Team 1.
            team_2_color: RGB color for Team 2.
        """
        self.default_player_team_id = 1
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color

    def draw_batch(
        self,
        video_frames: List[Any],
        tracks_batch: List[Dict[int, Dict[str, Any]]],
        player_assignment_batch: List[Dict[int, int]] = None,
    ) -> List[Any]:
        """
        Draw player tracks on a batch of frames.

        Args:
            video_frames: List of frames (NumPy arrays or images).
            tracks_batch: List of dicts mapping track_id -> player data (must include "bbox").
            player_assignment_batch: List of dicts mapping track_id -> team_id. If None, alternating team assignment is used.

        Returns:
            List of frames with player tracks drawn.
        """
        if not (len(video_frames) == len(tracks_batch)):
            raise ValueError("Input lists must have the same length.")

        output_frames = []
        append_frame = output_frames.append

        for frame_idx, (frame, player_dict) in enumerate(
            zip(video_frames, tracks_batch)
        ):
            frame_copy = frame.copy()

            # Get player assignment for this frame
            if player_assignment_batch and frame_idx < len(player_assignment_batch):
                assignment = player_assignment_batch[frame_idx]
            else:
                # Default: alternating team assignment based on track_id
                assignment = {}

            for track_id, player in player_dict.items():
                bbox = player.get("bbox")
                if bbox is None:
                    continue

                if track_id in assignment:
                    team_id = assignment[track_id]
                else:
                    team_id = (
                        1 if track_id % 2 == 1 else 2
                    )  # Odd IDs = team 1, Even IDs = team 2

                color = self.team_1_color if team_id == 1 else self.team_2_color

                frame_copy = draw_ellipse(frame_copy, bbox, color, track_id)

            append_frame(frame_copy)

        return output_frames
