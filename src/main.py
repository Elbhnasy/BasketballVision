from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer
from team_assigner import TeamAssigner

import logging

logging.basicConfig(level=logging.WARNING)  # Reduced logging level
logger = logging.getLogger(__name__)


def main():
    # Configuration
    video_path = (
        "/home/elbahnasy/CodingWorkspace/BasketballVision/src/videos/input/video_1.mp4"
    )
    output_path = (
        "/home/elbahnasy/CodingWorkspace/BasketballVision/src/videos/output/video_1.mp4"
    )

    players_model_path = (
        "/home/elbahnasy/CodingWorkspace/BasketballVision/src/models/player_detector.pt"
    )
    ball_model_path = "/home/elbahnasy/CodingWorkspace/BasketballVision/src/models/ball_detector_model.pt"
    player_stub_path = "/home/elbahnasy/CodingWorkspace/BasketballVision/src/stubs/player_track_stub.pkl"
    ball_stub_path = (
        "/home/elbahnasy/CodingWorkspace/BasketballVision/src/stubs/ball_track_stub.pkl"
    )
    team_stub_path = (
        "/home/elbahnasy/CodingWorkspace/BasketballVision/src/stubs/team_assignment_stub.pkl"
    )

    # Load video frames
    logger.info("Loading video frames...")
    frames = read_video(video_path)
    logger.info(f"Successfully loaded {len(frames)} frames")

    # Initialize trackers
    logger.info("Initializing trackers...")
    player_tracker = PlayerTracker(model_path=players_model_path, device="cpu")
    ball_tracker = BallTracker(model_path=ball_model_path, device="cpu")
    logger.info("Trackers initialized.")

    # Get player tracks with caching
    logger.info("Processing player tracks...")
    player_tracks = player_tracker.get_object_tracks(
        frames, read_from_stub=True, stub_path=player_stub_path
    )
    logger.info("Player tracks processed.")

    # Initialize team assigner and assign teams
    logger.info("Initializing team assigner...")
    team_assigner = TeamAssigner(
        team_1_class_name="white shirt",  
        team_2_class_name="blue shirt"
    )
    # Override device to CPU
    team_assigner.device = "cpu"
    logger.info("Team assigner initialized (forced CPU mode).")


    logger.info("Assigning players to teams...")
    team_assignments = team_assigner.get_player_team_across_frames(
        video_frames=frames,
        player_tracks=player_tracks,
        read_from_stub=True,
        stub_path=team_stub_path
    )
    logger.info("Team assignments completed.")

    # Get ball tracks with caching
    logger.info("Processing ball tracks...")
    ball_tracks = ball_tracker.get_object_tracks(
        frames, read_from_stub=True, stub_path=ball_stub_path
    )
    logger.info("Ball tracks processed.")

    # Process ball tracks - remove wrong detections and interpolate
    logger.info("Processing ball tracks: removing wrong detections...")
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    logger.info("Processing ball tracks: interpolating missing detections...")
    ball_tracks = ball_tracker.interpolate_missing_detections(ball_tracks)
    logger.info("Ball track processing completed.")

    # Initialize drawers
    logger.info("Initializing track drawers...")
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    logger.info("Track drawers initialized.")

    # Draw player tracks on frames with team assignments
    logger.info("Drawing player tracks on frames...")
    annotated_frames = player_tracks_drawer.draw_batch(
        video_frames=frames, 
        tracks_batch=player_tracks,
        player_assignment_batch=team_assignments  # Pass team assignments to drawer
    )
    logger.info("Player tracks drawn.")

    # Draw ball tracks on frames
    logger.info("Drawing ball tracks on frames...")
    annotated_frames = ball_tracks_drawer.draw_batch(
        video_frames=annotated_frames, tracks_batch=ball_tracks
    )
    logger.info("Ball tracks drawn.")

    # Save the final video
    logger.info("Saving annotated video...")
    save_video(annotated_frames, output_path)
    logger.info(f"Processing complete. Video saved to: {output_path}")


if __name__ == "__main__":
    main()

