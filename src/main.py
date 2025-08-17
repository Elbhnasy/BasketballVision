from utils import read_video, save_video
from trackers import PlayerTracker

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    video_path = "/home/elbahnasy/CodingWorkspace/BasketballVision/src/videos/input/video_1.mp4"
    output_path = "/home/elbahnasy/CodingWorkspace/BasketballVision/src/videos/output/video_1.mp4"
    
    model_path = "/home/elbahnasy/CodingWorkspace/BasketballVision/src/models/player_detector.pt"
    stub_path = "/home/elbahnasy/CodingWorkspace/BasketballVision/src/stubs/track_stub.pkl"

    # Read video frames
    frames = read_video(video_path)

    # Initialize player tracker (force CPU usage due to GPU compatibility)
    tracker = PlayerTracker(model_path=model_path, device='cpu')

    # Get object tracks
    player_tracks = tracker.get_object_tracks(frames,
                                        read_from_stub=True,
                                        stub_path=stub_path)
    logging.info(f"Player tracks: {player_tracks}")

    save_video(frames, output_path)

if __name__ == "__main__":
    main()