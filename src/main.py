from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import (
    PlayerTracksDrawer,
    BallTracksDrawer
)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    video_path = (
        "/home/elbahnasy/CodingWorkspace/BasketballVision/src/videos/input/video_1.mp4"
    )
    output_path = (
        "/home/elbahnasy/CodingWorkspace/BasketballVision/src/videos/output/video_1.mp4"
    )

    players_model_path = (
        "/home/elbahnasy/CodingWorkspace/BasketballVision/src/models/player_detector.pt"
    )
    ball_model_path = (
        "/home/elbahnasy/CodingWorkspace/BasketballVision/src/models/ball_detector_model.pt"
    )
    player_stub_path = (
        "/home/elbahnasy/CodingWorkspace/BasketballVision/src/stubs/player_track_stub.pkl"
    )
    ball_stub_path = (
        "/home/elbahnasy/CodingWorkspace/BasketballVision/src/stubs/ball_track_stub.pkl"
    )

    frames = read_video(video_path)
    logger.info("Loaded %d frames", len(frames))

    player_tracker = PlayerTracker(model_path=players_model_path, device="cpu")
    ball_tracker = BallTracker(model_path=ball_model_path, device="cpu")

    logger.info("Getting player tracks...")
    player_tracks = player_tracker.get_object_tracks(
        frames, read_from_stub=True, stub_path=player_stub_path
    )

    ball_tracks = ball_tracker.get_object_tracks(
        frames, read_from_stub=False, stub_path=ball_stub_path
    )
    logger.info("Tracks loaded.")

    logger.info("Drawing player tracks on frames (batch mode)...")
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()


    annotated_frames = player_tracks_drawer.draw_batch(
        video_frames=frames, tracks_batch=player_tracks
    )

    
    annotated_frames = ball_tracks_drawer.draw_batch(
        video_frames=annotated_frames, tracks_batch=ball_tracks
    )
    logger.info("Drawing complete.")

    save_video(annotated_frames, output_path)
    logger.info("Processing complete.")


if __name__ == "__main__":
    main()
