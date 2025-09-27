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
    
    def test_numpy_array_operations(self):
        """Test basic numpy operations for video processing."""
        # Test array creation and manipulation
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        assert frame.shape == (240, 320, 3)
        assert frame.sum() == 0
        
        # Test frame modification
        frame[100:140, 160:200] = [255, 0, 0]  # Red rectangle
        assert frame[120, 180, 0] == 255  # Red channel
        assert frame[120, 180, 1] == 0    # Green channel
        assert frame[120, 180, 2] == 0    # Blue channel
    
    def test_frame_copying_and_modification(self):
        """Test frame copying to avoid modifying originals."""
        original = np.ones((50, 50, 3), dtype=np.uint8) * 100
        copy = original.copy()
        
        # Modify the copy
        copy[25, 25] = [255, 255, 255]
        
        # Original should be unchanged
        assert np.all(original[25, 25] == [100, 100, 100])
        assert np.all(copy[25, 25] == [255, 255, 255])
    
    def test_batch_processing_simulation(self):
        """Test batch processing of frames."""
        batch_size = 5
        total_frames = 12
        
        # Simulate processing frames in batches
        frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) 
                 for _ in range(total_frames)]
        
        processed_frames = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            # Simulate some processing (just copy here)
            processed_batch = [frame.copy() for frame in batch]
            processed_frames.extend(processed_batch)
        
        assert len(processed_frames) == total_frames
        assert all(frame.shape == (100, 100, 3) for frame in processed_frames)


class TestImportValidation:
    """Tests to validate that required modules can be imported."""
    
    def test_import_trackers(self):
        """Test that tracker modules can be imported."""
        try:
            from trackers.ball_tracker import BallTracker
            from trackers.player_tracker import PlayerTracker
            assert BallTracker is not None
            assert PlayerTracker is not None
        except ImportError as e:
            pytest.fail(f"Failed to import tracker modules: {e}")
    
    def test_import_drawers(self):
        """Test that drawer modules can be imported."""
        try:
            from drawers.ball_tracks_drawer import BallTracksDrawer
            from drawers.player_tracks_drawer import PlayerTracksDrawer
            assert BallTracksDrawer is not None
            assert PlayerTracksDrawer is not None
        except ImportError as e:
            pytest.fail(f"Failed to import drawer modules: {e}")
    
    def test_import_utils(self):
        """Test that utility modules can be imported."""
        try:
            from utils.bbox_utils import get_bbox_center, get_bbox_width, get_foot_position
            from utils.stubs_utils import save_stub, read_stub
            assert callable(get_bbox_center)
            assert callable(get_bbox_width)
            assert callable(get_foot_position)
            assert callable(save_stub)
            assert callable(read_stub)
        except ImportError as e:
            pytest.fail(f"Failed to import utility modules: {e}")
    
    def test_external_dependencies(self):
        """Test that external dependencies are available."""
        try:
            import torch
            import pandas as pd
            from ultralytics import YOLO
            import supervision as sv
            
            # Basic functionality tests
            assert hasattr(torch, 'save')
            assert hasattr(torch, 'load')
            assert hasattr(pd, 'DataFrame')
            assert YOLO is not None
            assert hasattr(sv, 'ByteTrack')
        except ImportError as e:
            pytest.fail(f"Required external dependency not available: {e}")
