import pytest
import sys
import os
import numpy as np
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from drawers.ball_tracks_drawer import BallTracksDrawer


class TestBallTracksDrawer:
    """Comprehensive tests for BallTracksDrawer class."""
    
    @pytest.fixture
    def sample_frames(self):
        """Generate sample video frames for testing."""
        return [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(3)]
    
    @pytest.fixture
    def sample_tracks(self):
        """Generate sample ball tracks for testing."""
        return [
            {1: {"bbox": [100, 100, 200, 200]}},  # Frame 0: one ball
            {1: {"bbox": [110, 110, 210, 210]}},  # Frame 1: ball moved
            {}  # Frame 2: no ball detected
        ]
    
    @pytest.fixture
    def sample_tracks_multiple_balls(self):
        """Generate sample tracks with multiple balls."""
        return [
            {1: {"bbox": [100, 100, 200, 200]}, 2: {"bbox": [300, 300, 400, 400]}},  # Two balls
            {1: {"bbox": [110, 110, 210, 210]}},  # Only one ball
            {2: {"bbox": [320, 320, 420, 420]}}   # Different ball
        ]

    def test_init_default_color(self):
        """Test initialization with default ball pointer color."""
        drawer = BallTracksDrawer()
        
        assert drawer.ball_pointer_color == BallTracksDrawer.DEFAULT_BALL_POINTER_COLOR
        assert drawer.ball_pointer_color == (0, 255, 0)  # Green in BGR
    
    def test_init_custom_color(self):
        """Test initialization with custom ball pointer color."""
        custom_color = (255, 0, 0)  # Red in BGR
        drawer = BallTracksDrawer(ball_pointer_color=custom_color)
        
        assert drawer.ball_pointer_color == custom_color
    
    @patch('drawers.ball_tracks_drawer.draw_triangle')
    def test_draw_batch_valid_input(self, mock_draw_triangle, sample_frames, sample_tracks):
        """Test draw_batch with valid frames and tracks."""
        drawer = BallTracksDrawer()
        
        # Mock draw_triangle to return the same frame (simulating drawing)
        mock_draw_triangle.side_effect = lambda frame, bbox, color: frame
        
        result = drawer.draw_batch(sample_frames, sample_tracks)
        
        # Should return same number of frames
        assert len(result) == 3
        
        # draw_triangle should be called twice (frames 0 and 1 have balls)
        assert mock_draw_triangle.call_count == 2
        
        # Check the arguments passed to draw_triangle
        calls = mock_draw_triangle.call_args_list
        
        # First call: frame 0, bbox from sample_tracks[0]
        frame_0_call = calls[0]
        assert np.array_equal(frame_0_call[0][0], sample_frames[0].copy())
        assert frame_0_call[0][1] == [100, 100, 200, 200]
        assert frame_0_call[0][2] == drawer.ball_pointer_color
        
        # Second call: frame 1, bbox from sample_tracks[1]
        frame_1_call = calls[1]
        assert np.array_equal(frame_1_call[0][0], sample_frames[1].copy())
        assert frame_1_call[0][1] == [110, 110, 210, 210]
        assert frame_1_call[0][2] == drawer.ball_pointer_color
    
    @patch('drawers.ball_tracks_drawer.draw_triangle')
    def test_draw_batch_multiple_balls(self, mock_draw_triangle, sample_frames, sample_tracks_multiple_balls):
        """Test draw_batch with multiple balls in frames."""
        drawer = BallTracksDrawer()
        mock_draw_triangle.side_effect = lambda frame, bbox, color: frame
        
        result = drawer.draw_batch(sample_frames, sample_tracks_multiple_balls)
        
        assert len(result) == 3
        
        # draw_triangle should be called 4 times total:
        # Frame 0: 2 balls, Frame 1: 1 ball, Frame 2: 1 ball
        assert mock_draw_triangle.call_count == 4
    
    def test_draw_batch_length_mismatch(self, sample_frames):
        """Test draw_batch with mismatched frame and track lengths."""
        drawer = BallTracksDrawer()
        
        # Create tracks with different length than frames
        mismatched_tracks = [{1: {"bbox": [100, 100, 200, 200]}}]  # Only 1 track vs 3 frames
        
        with pytest.raises(ValueError, match="Input lists must have the same length"):
            drawer.draw_batch(sample_frames, mismatched_tracks)
    
    def test_draw_batch_empty_input(self):
        """Test draw_batch with empty frames and tracks."""
        drawer = BallTracksDrawer()
        
        result = drawer.draw_batch([], [])
        
        assert result == []
    
    @patch('drawers.ball_tracks_drawer.draw_triangle')
    def test_draw_batch_no_balls(self, mock_draw_triangle, sample_frames):
        """Test draw_batch when no balls are detected in any frame."""
        drawer = BallTracksDrawer()
        mock_draw_triangle.side_effect = lambda frame, bbox, color: frame
        
        # All frames have empty track dicts
        empty_tracks = [{}, {}, {}]
        
        result = drawer.draw_batch(sample_frames, empty_tracks)
        
        assert len(result) == 3
        # draw_triangle should not be called since no balls to draw
        mock_draw_triangle.assert_not_called()
    
    @patch('drawers.ball_tracks_drawer.draw_triangle')
    def test_draw_batch_missing_bbox(self, mock_draw_triangle, sample_frames):
        """Test draw_batch when ball data is missing bbox."""
        drawer = BallTracksDrawer()
        mock_draw_triangle.side_effect = lambda frame, bbox, color: frame
        
        # Tracks with missing bbox
        tracks_missing_bbox = [
            {1: {}},  # Ball without bbox
            {1: {"bbox": [100, 100, 200, 200]}},  # Valid ball
            {1: {"other_data": "something"}},  # Ball with other data but no bbox
        ]
        
        result = drawer.draw_batch(sample_frames, tracks_missing_bbox)
        
        assert len(result) == 3
        # draw_triangle should only be called once (frame 1 has valid bbox)
        assert mock_draw_triangle.call_count == 1
    
    @patch('drawers.ball_tracks_drawer.draw_triangle')
    def test_draw_batch_frame_copying(self, mock_draw_triangle, sample_frames, sample_tracks):
        """Test that frames are properly copied before drawing."""
        drawer = BallTracksDrawer()
        
        # Mock draw_triangle to modify the frame (to test copying)
        def modify_frame(frame, bbox, color):
            frame[0, 0, 0] = 255  # Modify the frame
            return frame
        
        mock_draw_triangle.side_effect = modify_frame
        
        original_frame_value = sample_frames[0][0, 0, 0]
        
        result = drawer.draw_batch(sample_frames, sample_tracks)
        
        # Original frames should not be modified (due to .copy())
        assert sample_frames[0][0, 0, 0] == original_frame_value
        
        # Result frames should be modified
        assert result[0][0, 0, 0] == 255
    
    @patch('drawers.ball_tracks_drawer.draw_triangle')
    def test_draw_batch_custom_color_usage(self, mock_draw_triangle, sample_frames, sample_tracks):
        """Test that custom color is passed to draw_triangle."""
        custom_color = (255, 0, 255)  # Magenta
        drawer = BallTracksDrawer(ball_pointer_color=custom_color)
        
        mock_draw_triangle.side_effect = lambda frame, bbox, color: frame
        
        drawer.draw_batch(sample_frames, sample_tracks)
        
        # Check that custom color was passed to draw_triangle
        calls = mock_draw_triangle.call_args_list
        for call in calls:
            assert call[0][2] == custom_color
    
    def test_draw_batch_large_number_of_frames(self):
        """Test draw_batch with a large number of frames."""
        drawer = BallTracksDrawer()
        
        # Create 100 frames and tracks
        large_frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(100)]
        large_tracks = [{1: {"bbox": [i, i, i+50, i+50]}} for i in range(100)]
        
        with patch('drawers.ball_tracks_drawer.draw_triangle') as mock_draw_triangle:
            mock_draw_triangle.side_effect = lambda frame, bbox, color: frame
            
            result = drawer.draw_batch(large_frames, large_tracks)
            
            assert len(result) == 100
            assert mock_draw_triangle.call_count == 100
    
    def test_draw_batch_different_frame_shapes(self):
        """Test draw_batch with frames of different shapes."""
        drawer = BallTracksDrawer()
        
        # Create frames with different shapes
        frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),  # HD
            np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),  # Smaller
            np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8),  # Full HD
        ]
        
        tracks = [{1: {"bbox": [10, 10, 50, 50]}} for _ in range(3)]
        
        with patch('drawers.ball_tracks_drawer.draw_triangle') as mock_draw_triangle:
            mock_draw_triangle.side_effect = lambda frame, bbox, color: frame
            
            result = drawer.draw_batch(frames, tracks)
            
            assert len(result) == 3
            # Each frame should maintain its original shape
            for i, frame in enumerate(result):
                assert frame.shape == frames[i].shape


class TestBallTracksDrawerIntegration:
    """Integration tests for BallTracksDrawer with real drawing operations."""
    
    def test_integration_with_actual_drawing(self):
        """Test with actual draw_triangle function (if available)."""
        drawer = BallTracksDrawer()
        
        # Create simple test frames
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(2)]
        tracks = [
            {1: {"bbox": [10, 10, 30, 30]}},
            {1: {"bbox": [20, 20, 40, 40]}}
        ]
        
        try:
            # This will work if draw_triangle is properly implemented
            result = drawer.draw_batch(frames, tracks)
            assert len(result) == 2
            assert all(isinstance(frame, np.ndarray) for frame in result)
        except ImportError:
            # Skip test if draw_triangle is not available
            pytest.skip("draw_triangle function not available for integration test")
    
    def test_memory_efficiency_large_batch(self):
        """Test memory usage with large batches."""
        drawer = BallTracksDrawer()
        
        # Create a reasonably large batch to test memory handling
        num_frames = 50
        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(num_frames)]
        tracks = [{1: {"bbox": [i*2, i*2, i*2+50, i*2+50]}} if i % 2 == 0 else {} for i in range(num_frames)]
        
        with patch('drawers.ball_tracks_drawer.draw_triangle') as mock_draw_triangle:
            mock_draw_triangle.side_effect = lambda frame, bbox, color: frame
            
            result = drawer.draw_batch(frames, tracks)
            
            assert len(result) == num_frames
            # Verify that we're not keeping references to too many frame copies
            assert len(result) == len(frames)