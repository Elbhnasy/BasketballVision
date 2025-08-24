from utils import read_video, save_video
from trackers import PlayerTracker
from drawers import PlayerTracksDrawer

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    video_path = "/home/elbahnasy/CodingWorkspace/BasketballVision/src/videos/input/video_1.mp4"
    output_path = "/home/elbahnasy/CodingWorkspace/BasketballVision/src/videos/output/video_1.mp4"
    
    model_path = "/home/elbahnasy/CodingWorkspace/BasketballVision/src/models/player_detector.pt"
    stub_path = "/home/elbahnasy/CodingWorkspace/BasketballVision/src/stubs/track_stub.pkl"

    logger.info("Reading video from %s", video_path)
    frames = read_video(video_path)
    logger.info("Loaded %d frames", len(frames))

    logger.info("Initializing player tracker...")
    tracker = PlayerTracker(model_path=model_path, device='cpu')

    logger.info("Getting player tracks...")
    player_tracks = tracker.get_object_tracks(
        frames,
        read_from_stub=True,
        stub_path=stub_path
    )
    logger.info("Tracks loaded.")

    logger.info("Drawing player tracks on frames (batch mode)...")
    player_tracks_drawer = PlayerTracksDrawer()
    annotated_frames = player_tracks_drawer.draw_batch(
        video_frames=frames,
        tracks_batch=player_tracks
    )

    save_video(annotated_frames, output_path)
    logger.info("Processing complete.")

if __name__ == "__main__":
    main()
