from utils import read_video, save_video

def main():
    video_path = "/home/elbahnasy/CodingWorkspace/BasketballVision/src/videos/input/video_1.mp4"
    output_path = "/home/elbahnasy/CodingWorkspace/BasketballVision/src/videos/output/video_1.mp4"
    frames = read_video(video_path)
    save_video(frames, output_path)

if __name__ == "__main__":
    main()