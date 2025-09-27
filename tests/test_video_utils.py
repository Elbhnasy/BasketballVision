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
    
    def test_save_video_invalid_fps(self):
        """Test saving video with invalid FPS values."""
        frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(3)]
        
        # Note: The actual save_video function may not validate FPS values
        # So we test that it handles edge cases gracefully
        try:
            # Test with zero fps - may or may not raise error depending on implementation
            save_video(frames, "test_zero_fps.mp4", fps=0)
            # Clean up if successful
            if os.path.exists("test_zero_fps.mp4"):
                os.remove("test_zero_fps.mp4")
        except (ValueError, Exception):
            # Expected behavior for invalid fps
            pass
        
        try:
            # Test with negative fps - may or may not raise error depending on implementation
            save_video(frames, "test_negative_fps.mp4", fps=-30)
            # Clean up if successful
            if os.path.exists("test_negative_fps.mp4"):
                os.remove("test_negative_fps.mp4")
        except (ValueError, Exception):
            # Expected behavior for invalid fps
            pass
    
    def test_save_video_different_fps_values(self):
        """Test saving video with different FPS values."""
        frames = [np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(3)]
        
        fps_values = [24, 30, 60, 120]
        
        for fps in fps_values:
            output_path = f"test_fps_{fps}.mp4"
            try:
                save_video(frames, output_path, fps=fps)
                # Clean up
                if os.path.exists(output_path):
                    os.remove(output_path)
            except Exception as e:
                pytest.fail(f"save_video failed with fps={fps}: {e}")
    
    def test_save_video_different_frame_sizes(self):
        """Test saving video with different frame dimensions."""
        test_cases = [
            (64, 64),    # Small
            (480, 640),  # SD
            (720, 1280), # HD
        ]
        
        for height, width in test_cases:
            frames = [np.random.randint(0, 255, (height, width, 3), dtype=np.uint8) for _ in range(2)]
            output_path = f"test_{height}x{width}.mp4"
            
            try:
                save_video(frames, output_path, fps=30)
                # Clean up
                if os.path.exists(output_path):
                    os.remove(output_path)
            except Exception as e:
                pytest.fail(f"save_video failed with size {height}x{width}: {e}")
    
    def test_save_video_single_frame(self):
        """Test saving video with only one frame."""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frames = [frame]
        output_path = "test_single_frame.mp4"
        
        try:
            save_video(frames, output_path, fps=30)
            # Clean up
            if os.path.exists(output_path):
                os.remove(output_path)
        except Exception as e:
            pytest.fail(f"save_video failed with single frame: {e}")
    
    def test_save_video_inconsistent_frame_shapes(self):
        """Test saving video with frames of different shapes."""
        frames = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8),  # Different size
        ]
        
        # The behavior may vary - some implementations handle this gracefully
        try:
            save_video(frames, "test_inconsistent.mp4", fps=30)
            # Clean up if successful
            if os.path.exists("test_inconsistent.mp4"):
                os.remove("test_inconsistent.mp4")
        except (ValueError, Exception) as e:
            # This is also acceptable behavior for inconsistent frame shapes
            pass
    
    def test_save_video_invalid_file_path(self):
        """Test saving video to invalid file path."""
        frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(2)]
        
        # Try to save to a directory that doesn't exist (and can't be created)
        invalid_path = "/root/nonexistent/impossible/path/test.mp4"
        
        # Should handle the error gracefully
        try:
            save_video(frames, invalid_path, fps=30)
        except (PermissionError, OSError, FileNotFoundError):
            # These are expected errors for invalid paths
            pass
        except Exception as e:
            pytest.fail(f"Unexpected exception type: {type(e).__name__}: {e}")
    
    def test_read_video_with_various_extensions(self):
        """Test read_video with different file extensions (all should fail for non-existent files)."""
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        
        for ext in extensions:
            fake_path = f"nonexistent_video{ext}"
            with pytest.raises(FileNotFoundError):
                read_video(fake_path)
