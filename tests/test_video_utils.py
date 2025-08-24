import pytest
import sys
import os
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.video_utils import read_video, save_video


class TestVideoUtils:
    """Basic tests for video utilities."""
    
    def test_read_video_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent video."""
        with pytest.raises(FileNotFoundError):
            read_video("non_existent_video.mp4")
    
    def test_read_video_invalid_path(self):
        """Test that FileNotFoundError is raised for invalid path."""
        with pytest.raises(FileNotFoundError):
            read_video("")
    
    def test_save_video_with_valid_frames(self):
        """Test saving video with valid frame data."""
        # Create dummy frames (100x100 RGB)
        frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(5)]
        output_path = "test_output.mp4"
        
        try:
            # This should not raise an exception
            save_video(frames, output_path, fps=30)
            # Clean up
            if os.path.exists(output_path):
                os.remove(output_path)
        except Exception as e:
            pytest.fail(f"save_video raised an exception: {e}")
    
    def test_save_video_empty_frames(self):
        """Test saving video with empty frames list."""
        with pytest.raises((ValueError, IndexError)):
            save_video([], "empty_output.mp4", fps=30)
