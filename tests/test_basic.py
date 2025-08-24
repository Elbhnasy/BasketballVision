import pytest
import sys
import os
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestBasicImports:
    """Basic tests to ensure core modules can be imported."""
    
    def test_import_video_utils(self):
        """Test that video_utils module can be imported."""
        try:
            from utils.video_utils import read_video, save_video
            assert callable(read_video)
            assert callable(save_video)
        except ImportError as e:
            pytest.fail(f"Failed to import video_utils: {e}")
    
    def test_import_main(self):
        """Test that main module can be imported."""
        try:
            import main
        except ImportError as e:
            pytest.fail(f"Failed to import main: {e}")
    
    def test_numpy_available(self):
        """Test that numpy is available and working."""
        arr = np.array([1, 2, 3])
        assert arr.shape == (3,)
        assert arr.dtype == np.int64 or arr.dtype == np.int32
    
    def test_opencv_available(self):
        """Test that OpenCV is available."""
        try:
            import cv2
            # Test basic OpenCV functionality
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            assert gray.shape == (100, 100)
        except ImportError as e:
            pytest.fail(f"OpenCV not available: {e}")


class TestBasicFunctionality:
    """Basic functionality tests."""
    
    def test_create_dummy_video_frame(self):
        """Test creating a basic video frame."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8
    
    def test_frame_list_operations(self):
        """Test basic operations on frame lists."""
        frames = []
        for i in range(3):
            frame = np.ones((100, 100, 3), dtype=np.uint8) * i * 50
            frames.append(frame)
        
        assert len(frames) == 3
        assert frames[0].max() == 0
        assert frames[1].max() == 50
        assert frames[2].max() == 100
