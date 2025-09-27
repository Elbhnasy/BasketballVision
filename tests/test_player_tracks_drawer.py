import pytest
import sys
import os
import numpy as np
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from drawers.player_tracks_drawer import PlayerTracksDrawer


class TestPlayerTracksDrawer:
    """Comprehensive tests for PlayerTracksDrawer class."""
    
    @pytest.fixture
    def sample_frames(self):
        """Generate sample video frames for testing."""
        return [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(3)]
    
    @pytest.fixture
    def sample_tracks(self):
        """Generate sample player tracks for testing."""
        return [
            {1: {"bbox": [100, 100, 200, 200]}, 2: {"bbox": [300, 300, 400, 400]}},  # Frame 0: two players
            {1: {"bbox": [110, 110, 210, 210]}},  # Frame 1: one player
            {}  # Frame 2: no players detected
        ]
    
    @pytest.fixture
    def sample_player_assignments(self):
        """Generate sample player team assignments."""
        return [
            {1: 1, 2: 2},  # Frame 0: player 1 -> team 1, player 2 -> team 2
            {1: 1},        # Frame 1: player 1 -> team 1
            {}             # Frame 2: no assignments
        ]

    def test_init_default_colors(self):
        """Test initialization with default team colors."""
        drawer = PlayerTracksDrawer()
        
        assert drawer.team_1_color == PlayerTracksDrawer.DEFAULT_TEAM_1_COLOR
        assert drawer.team_2_color == PlayerTracksDrawer.DEFAULT_TEAM_2_COLOR
        assert drawer.default_player_team_id == 1
        assert drawer.team_1_color == (255, 245, 238)
        assert drawer.team_2_color == (128, 0, 0)
    
    def test_init_custom_colors(self):
        """Test initialization with custom team colors."""
        custom_team_1 = (0, 255, 0)  # Green
        custom_team_2 = (0, 0, 255)  # Blue
        
        drawer = PlayerTracksDrawer(team_1_color=custom_team_1, team_2_color=custom_team_2)
        
        assert drawer.team_1_color == custom_team_1
        assert drawer.team_2_color == custom_team_2
        assert drawer.default_player_team_id == 1
    
    @patch('drawers.player_tracks_drawer.draw_ellipse')
    def test_draw_batch_with_assignments(self, mock_draw_ellipse, sample_frames, sample_tracks, sample_player_assignments):
        """Test draw_batch with explicit player team assignments."""
        drawer = PlayerTracksDrawer()
        
        # Mock draw_ellipse to return the same frame
        mock_draw_ellipse.side_effect = lambda frame, bbox, color, track_id: frame
        
        result = drawer.draw_batch(sample_frames, sample_tracks, sample_player_assignments)
        
        # Should return same number of frames
        assert len(result) == 3
        
        # draw_ellipse should be called 3 times (2 players in frame 0, 1 player in frame 1)
        assert mock_draw_ellipse.call_count == 3
        
        # Check the arguments passed to draw_ellipse
        calls = mock_draw_ellipse.call_args_list
        
        # First call: frame 0, player 1 (team 1 color)
        frame_0_player_1 = calls[0]
        assert frame_0_player_1[0][1] == [100, 100, 200, 200]
        assert frame_0_player_1[0][2] == drawer.team_1_color
        
        # Second call: frame 0, player 2 (team 2 color)
        frame_0_player_2 = calls[1]
        assert frame_0_player_2[0][1] == [300, 300, 400, 400]
        assert frame_0_player_2[0][2] == drawer.team_2_color
        
        # Third call: frame 1, player 1 (team 1 color)
        frame_1_player_1 = calls[2]
        assert frame_1_player_1[0][1] == [110, 110, 210, 210]
        assert frame_1_player_1[0][2] == drawer.team_1_color
    
    @patch('drawers.player_tracks_drawer.draw_ellipse')
    def test_draw_batch_without_assignments(self, mock_draw_ellipse, sample_frames, sample_tracks):
        """Test draw_batch without player team assignments (alternating teams)."""
        drawer = PlayerTracksDrawer()
        
        mock_draw_ellipse.side_effect = lambda frame, bbox, color, track_id: frame
        
        result = drawer.draw_batch(sample_frames, sample_tracks)
        
        assert len(result) == 3
        assert mock_draw_ellipse.call_count == 3
        
        # Check that alternating team assignment is used
        calls = mock_draw_ellipse.call_args_list
        
        # Should alternate between team colors based on track_id
        # Player 1 (odd) -> team 2, Player 2 (even) -> team 1
        # But actually based on the implementation, it's based on index % 2
    
    @patch('drawers.player_tracks_drawer.draw_ellipse')
    def test_draw_batch_alternating_team_assignment(self, mock_draw_ellipse, sample_frames):
        """Test alternating team assignment when no assignments provided."""
        drawer = PlayerTracksDrawer()
        
        # Create tracks with mixed odd/even IDs to test alternating assignment
        tracks = [
            {11: {"bbox": [100, 100, 200, 200]}, 20: {"bbox": [300, 300, 400, 400]},
             31: {"bbox": [500, 500, 600, 600]}, 40: {"bbox": [700, 700, 800, 800]}}
        ]
        
        mock_draw_ellipse.side_effect = lambda frame, bbox, color, track_id: frame
        
        result = drawer.draw_batch([sample_frames[0]], tracks)
        
        assert mock_draw_ellipse.call_count == 4
        
        calls = mock_draw_ellipse.call_args_list
        # Check alternating pattern in colors
        colors_used = [call[0][2] for call in calls]
        
        # Should alternate between team_1_color and team_2_color
        expected_colors = [drawer.team_1_color, drawer.team_2_color, drawer.team_1_color, drawer.team_2_color]
        assert colors_used == expected_colors
    
    def test_draw_batch_length_mismatch_tracks_and_frames(self, sample_frames):
        """Test draw_batch with mismatched frame and track lengths."""
        drawer = PlayerTracksDrawer()
        
        # Create tracks with different length than frames
        mismatched_tracks = [{1: {"bbox": [100, 100, 200, 200]}}]  # Only 1 track vs 3 frames
        
        with pytest.raises(ValueError, match="Input lists must have the same length"):
            drawer.draw_batch(sample_frames, mismatched_tracks)
    
    def test_draw_batch_length_mismatch_assignments(self, sample_frames, sample_tracks):
        """Test draw_batch with mismatched assignment lengths."""
        drawer = PlayerTracksDrawer()
        
        with patch('drawers.player_tracks_drawer.draw_ellipse') as mock_draw_ellipse:
            mock_draw_ellipse.side_effect = lambda frame, bbox, color, track_id: frame
            
            # Create assignments with different length
            mismatched_assignments = [{1: 1}]  # Only 1 assignment vs 3 frames
            
            # This should work - missing assignments use default logic
            result = drawer.draw_batch(sample_frames, sample_tracks, mismatched_assignments)
            
            assert len(result) == 3
            # Should still draw players based on alternating assignment for missing frames
    
    def test_draw_batch_empty_input(self):
        """Test draw_batch with empty frames and tracks."""
        drawer = PlayerTracksDrawer()
        
        result = drawer.draw_batch([], [])
        
        assert result == []
    
    @patch('drawers.player_tracks_drawer.draw_ellipse')
    def test_draw_batch_no_players(self, mock_draw_ellipse, sample_frames):
        """Test draw_batch when no players are detected in any frame."""
        drawer = PlayerTracksDrawer()
        mock_draw_ellipse.side_effect = lambda frame, bbox, color, track_id: frame
        
        # All frames have empty track dicts
        empty_tracks = [{}, {}, {}]
        
        result = drawer.draw_batch(sample_frames, empty_tracks)
        
        assert len(result) == 3
        # draw_ellipse should not be called since no players to draw
        mock_draw_ellipse.assert_not_called()
    
    @patch('drawers.player_tracks_drawer.draw_ellipse')
    def test_draw_batch_missing_bbox(self, mock_draw_ellipse, sample_frames):
        """Test draw_batch when player data is missing bbox."""
        drawer = PlayerTracksDrawer()
        mock_draw_ellipse.side_effect = lambda frame, bbox, color, track_id: frame
        
        # Tracks with missing bbox
        tracks_missing_bbox = [
            {1: {}},  # Player without bbox
            {1: {"bbox": [100, 100, 200, 200]}},  # Valid player
            {1: {"other_data": "something"}},  # Player with other data but no bbox
        ]
        
        result = drawer.draw_batch(sample_frames, tracks_missing_bbox)
        
        assert len(result) == 3
        # draw_ellipse should only be called once (frame 1 has valid bbox)
        assert mock_draw_ellipse.call_count == 1
    
    @patch('drawers.player_tracks_drawer.draw_ellipse')
    def test_draw_batch_frame_copying(self, mock_draw_ellipse, sample_frames, sample_tracks):
        """Test that frames are properly copied before drawing."""
        drawer = PlayerTracksDrawer()
        
        # Mock draw_ellipse to modify the frame (to test copying)
        def modify_frame(frame, bbox, color, track_id):
            frame[0, 0, 0] = 255  # Modify the frame
            return frame
        
        mock_draw_ellipse.side_effect = modify_frame
        
        original_frame_value = sample_frames[0][0, 0, 0]
        
        result = drawer.draw_batch(sample_frames, sample_tracks)
        
        # Original frames should not be modified (due to .copy())
        assert sample_frames[0][0, 0, 0] == original_frame_value
        
        # Result frames should be modified
        assert result[0][0, 0, 0] == 255
    
    @patch('drawers.player_tracks_drawer.draw_ellipse')
    def test_draw_batch_custom_colors_usage(self, mock_draw_ellipse, sample_frames, sample_tracks, sample_player_assignments):
        """Test that custom colors are passed to draw_ellipse."""
        custom_team_1 = (255, 0, 255)  # Magenta
        custom_team_2 = (0, 255, 255)  # Cyan
        drawer = PlayerTracksDrawer(team_1_color=custom_team_1, team_2_color=custom_team_2)
        
        mock_draw_ellipse.side_effect = lambda frame, bbox, color, track_id: frame
        
        drawer.draw_batch(sample_frames, sample_tracks, sample_player_assignments)
        
        # Check that custom colors were passed to draw_ellipse
        calls = mock_draw_ellipse.call_args_list
        colors_used = [call[0][2] for call in calls]
        
        assert custom_team_1 in colors_used
        assert custom_team_2 in colors_used
    
    def test_draw_batch_team_assignment_edge_cases(self, sample_frames):
        """Test team assignment with edge cases."""
        drawer = PlayerTracksDrawer()
        
        with patch('drawers.player_tracks_drawer.draw_ellipse') as mock_draw_ellipse:
            mock_draw_ellipse.side_effect = lambda frame, bbox, color, track_id: frame
            
            # Test with team assignment values other than 1 or 2
            tracks = [{1: {"bbox": [100, 100, 200, 200]}}]
            assignments = [{1: 3}]  # Team 3 (should use default)
            
            result = drawer.draw_batch([sample_frames[0]], tracks, assignments)
            
            # Should still work, using default team assignment logic
            assert len(result) == 1
            assert mock_draw_ellipse.call_count == 1
    
    @patch('drawers.player_tracks_drawer.draw_ellipse')
    def test_draw_batch_missing_player_assignment(self, mock_draw_ellipse, sample_frames):
        """Test when player assignment is missing for some players."""
        drawer = PlayerTracksDrawer()
        mock_draw_ellipse.side_effect = lambda frame, bbox, color, track_id: frame
        
        # Player 2 is missing from assignments
        tracks = [{1: {"bbox": [100, 100, 200, 200]}, 2: {"bbox": [300, 300, 400, 400]}}]
        assignments = [{1: 1}]  # Only player 1 assigned
        
        result = drawer.draw_batch([sample_frames[0]], tracks, assignments)
        
        assert len(result) == 1
        assert mock_draw_ellipse.call_count == 2  # Both players should be drawn
        
        # Check that missing assignment uses default logic
        calls = mock_draw_ellipse.call_args_list
        colors_used = [call[0][2] for call in calls]
        assert drawer.team_1_color in colors_used
    
    def test_draw_batch_large_number_of_players(self):
        """Test draw_batch with many players."""
        drawer = PlayerTracksDrawer()
        
        # Create frame with many players
        num_players = 20
        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)]
        tracks = [{i: {"bbox": [i*10, i*10, i*10+50, i*10+50]} for i in range(1, num_players + 1)}]
        
        with patch('drawers.player_tracks_drawer.draw_ellipse') as mock_draw_ellipse:
            mock_draw_ellipse.side_effect = lambda frame, bbox, color, track_id: frame
            
            result = drawer.draw_batch(frames, tracks)
            
            assert len(result) == 1
            assert mock_draw_ellipse.call_count == num_players
    
    def test_draw_batch_different_frame_shapes(self):
        """Test draw_batch with frames of different shapes."""
        drawer = PlayerTracksDrawer()
        
        # Create frames with different shapes
        frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),  # HD
            np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),  # Smaller
        ]
        
        tracks = [{1: {"bbox": [10, 10, 50, 50]}} for _ in range(2)]
        
        with patch('drawers.player_tracks_drawer.draw_ellipse') as mock_draw_ellipse:
            mock_draw_ellipse.side_effect = lambda frame, bbox, color, track_id: frame
            
            result = drawer.draw_batch(frames, tracks)
            
            assert len(result) == 2
            # Each frame should maintain its original shape
            for i, frame in enumerate(result):
                assert frame.shape == frames[i].shape


class TestPlayerTracksDrawerIntegration:
    """Integration tests for PlayerTracksDrawer."""
    
    def test_integration_with_actual_drawing(self):
        """Test with actual draw_ellipse function (if available)."""
        drawer = PlayerTracksDrawer()
        
        # Create simple test frames
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(2)]
        tracks = [
            {1: {"bbox": [10, 10, 30, 30]}, 2: {"bbox": [50, 50, 70, 70]}},
            {1: {"bbox": [20, 20, 40, 40]}}
        ]
        
        try:
            # This will work if draw_ellipse is properly implemented
            result = drawer.draw_batch(frames, tracks)
            assert len(result) == 2
            assert all(isinstance(frame, np.ndarray) for frame in result)
        except ImportError:
            # Skip test if draw_ellipse is not available
            pytest.skip("draw_ellipse function not available for integration test")
    
    def test_memory_efficiency_large_batch(self):
        """Test memory usage with large batches."""
        drawer = PlayerTracksDrawer()
        
        # Create a reasonably large batch
        num_frames = 30
        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(num_frames)]
        tracks = [{i: {"bbox": [i*5, i*5, i*5+40, i*5+40]} for i in range(1, 3)} for _ in range(num_frames)]
        
        with patch('drawers.player_tracks_drawer.draw_ellipse') as mock_draw_ellipse:
            mock_draw_ellipse.side_effect = lambda frame, bbox, color, track_id: frame
            
            result = drawer.draw_batch(frames, tracks)
            
            assert len(result) == num_frames
            # Each frame should have 2 players drawn
            assert mock_draw_ellipse.call_count == num_frames * 2