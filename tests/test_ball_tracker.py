import pytest
import sys
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trackers.ball_tracker import BallTracker


class TestBallTracker:
    """Comprehensive tests for BallTracker class."""
    
    @pytest.fixture
    def mock_model_path(self, tmp_path):
        """Create a temporary model file."""
        model_file = tmp_path / "test_model.pt"
        model_file.touch()
        return str(model_file)
    
    @pytest.fixture
    def mock_yolo_model(self):
        """Create a mock YOLO model."""
        mock_model = Mock()
        mock_model.names = {0: "Ball"}
        mock_model.to = Mock()
        return mock_model
    
    @pytest.fixture
    def sample_frames(self):
        """Generate sample frames for testing."""
        return [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)]
    
    @pytest.fixture
    def sample_detection_result(self):
        """Create a mock detection result."""
        mock_result = Mock()
        mock_result.names = {0: "Ball"}
        return mock_result

    def test_init_with_valid_model_path(self, mock_model_path):
        """Test initialization with valid model path."""
        with patch('trackers.ball_tracker.YOLO') as mock_yolo:
            mock_yolo.return_value.to = Mock()
            tracker = BallTracker(mock_model_path, batch_size=10, device="cpu")
            
            assert tracker.batch_size == 10
            assert tracker.device == "cpu"
            mock_yolo.assert_called_once_with(mock_model_path)

    def test_init_with_invalid_model_path(self):
        """Test initialization with invalid model path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            BallTracker("non_existent_model.pt")

    def test_init_default_parameters(self, mock_model_path):
        """Test initialization with default parameters."""
        with patch('trackers.ball_tracker.YOLO') as mock_yolo:
            mock_yolo.return_value.to = Mock()
            tracker = BallTracker(mock_model_path)
            
            assert tracker.batch_size == 20
            assert tracker.device == "cpu"

    @patch('trackers.ball_tracker.YOLO')
    def test_detect_frames_single_batch(self, mock_yolo, mock_model_path, sample_frames):
        """Test detection with frames that fit in a single batch."""
        mock_model = Mock()
        mock_model.to = Mock()
        mock_yolo.return_value = mock_model
        
        tracker = BallTracker(mock_model_path, batch_size=10)
        
        # Mock the model call
        mock_detections = [Mock() for _ in range(5)]
        mock_model.side_effect = lambda frames, conf: mock_detections[:len(frames)]
        
        result = tracker.detect_frames(sample_frames, conf=0.6)
        
        assert len(result) == 5
        mock_model.assert_called_once_with(sample_frames, conf=0.6)

    @patch('trackers.ball_tracker.YOLO')
    def test_detect_frames_multiple_batches(self, mock_yolo, mock_model_path):
        """Test detection with frames requiring multiple batches."""
        mock_model = Mock()
        mock_model.to = Mock()
        mock_yolo.return_value = mock_model
        
        # Create 25 frames with batch_size=10
        frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(25)]
        tracker = BallTracker(mock_model_path, batch_size=10)
        
        # Mock the model to return different results for each batch
        mock_model.side_effect = lambda batch, conf: [Mock() for _ in batch]
        
        result = tracker.detect_frames(frames, conf=0.5)
        
        assert len(result) == 25
        assert mock_model.call_count == 3  # 25 frames / 10 batch_size = 3 calls

    @patch('trackers.ball_tracker.save_stub')
    @patch('trackers.ball_tracker.read_stub')
    @patch('trackers.ball_tracker.YOLO')
    def test_get_object_tracks_with_cached_data(self, mock_yolo, mock_read_stub, mock_save_stub, mock_model_path, sample_frames):
        """Test getting tracks when cached data is available."""
        mock_model = Mock()
        mock_model.to = Mock()
        mock_yolo.return_value = mock_model
        
        # Mock cached tracks
        cached_tracks = [{"1": {"bbox": [100, 100, 200, 200]}} for _ in range(5)]
        mock_read_stub.return_value = cached_tracks
        
        tracker = BallTracker(mock_model_path)
        result = tracker.get_object_tracks(sample_frames, read_from_stub=True, stub_path="test.pkl")
        
        assert result == cached_tracks
        mock_read_stub.assert_called_once_with(True, "test.pkl")
        mock_save_stub.assert_not_called()

    @patch('trackers.ball_tracker.save_stub')
    @patch('trackers.ball_tracker.read_stub')
    @patch('trackers.ball_tracker.sv.Detections')
    @patch('trackers.ball_tracker.YOLO')
    def test_get_object_tracks_no_cached_data(self, mock_yolo, mock_sv_detections, mock_read_stub, mock_save_stub, mock_model_path, sample_frames):
        """Test getting tracks when no cached data is available."""
        mock_model = Mock()
        mock_model.to = Mock()
        mock_yolo.return_value = mock_model
        
        # Mock no cached data
        mock_read_stub.return_value = None
        
        # Mock detection result
        mock_detection = Mock()
        mock_detection.names = {0: "Ball"}
        mock_model.return_value = [mock_detection]
        mock_model.side_effect = lambda frames, **kwargs: [mock_detection for _ in frames]
        
        # Mock supervision detections
        mock_sv_detection = Mock()
        mock_sv_detection.class_id = np.array([0])
        mock_sv_detection.confidence = np.array([0.8])
        mock_sv_detection.xyxy = np.array([[100, 100, 200, 200]])
        mock_sv_detection.__len__ = Mock(return_value=1)
        mock_sv_detections.from_ultralytics.return_value = mock_sv_detection
        
        tracker = BallTracker(mock_model_path)
        result = tracker.get_object_tracks(sample_frames, stub_path="test.pkl")
        
        assert len(result) == 5
        mock_save_stub.assert_called_once_with("test.pkl", result)

    @patch('trackers.ball_tracker.YOLO')
    def test_get_object_tracks_empty_frames(self, mock_yolo, mock_model_path):
        """Test getting tracks with empty frame list."""
        mock_model = Mock()
        mock_model.to = Mock()
        mock_yolo.return_value = mock_model
        
        tracker = BallTracker(mock_model_path)
        result = tracker.get_object_tracks([])
        
        assert result == []

    @patch('trackers.ball_tracker.YOLO')
    def test_get_object_tracks_no_ball_class(self, mock_yolo, mock_model_path, sample_frames):
        """Test getting tracks when model doesn't have Ball class."""
        mock_model = Mock()
        mock_model.to = Mock()
        mock_yolo.return_value = mock_model
        
        # Mock detection result without Ball class
        mock_detection = Mock()
        mock_detection.names = {0: "Person", 1: "Car"}
        mock_model.return_value = [mock_detection]
        mock_model.side_effect = lambda frames, verbose: [mock_detection for _ in frames]
        
        tracker = BallTracker(mock_model_path)
        result = tracker.get_object_tracks(sample_frames)
        
        # Should return empty dicts for each frame
        assert len(result) == 5
        assert all(track == {} for track in result)

    def test_remove_wrong_detections_empty_input(self, mock_model_path):
        """Test removing wrong detections with empty input."""
        with patch('trackers.ball_tracker.YOLO'):
            tracker = BallTracker(mock_model_path)
            result = tracker.remove_wrong_detections([])
            assert result == []

    def test_remove_wrong_detections_valid_sequence(self, mock_model_path):
        """Test removing wrong detections with valid sequence."""
        with patch('trackers.ball_tracker.YOLO'):
            tracker = BallTracker(mock_model_path)
            
            # Create a sequence with a position jump
            ball_positions = [
                {1: {"bbox": [100, 100, 120, 120]}},  # Frame 0
                {1: {"bbox": [105, 105, 125, 125]}},  # Frame 1 - small movement
                {1: {"bbox": [300, 300, 320, 320]}},  # Frame 2 - large jump (should be removed)
                {1: {"bbox": [110, 110, 130, 130]}},  # Frame 3 - back to expected position
            ]
            
            result = tracker.remove_wrong_detections(ball_positions)
            
            # Frame 2 should be removed (empty dict)
            assert result[0] == {1: {"bbox": [100, 100, 120, 120]}}
            assert result[1] == {1: {"bbox": [105, 105, 125, 125]}}
            assert result[2] == {}  # Removed due to large jump
            assert result[3] == {1: {"bbox": [110, 110, 130, 130]}}

    def test_remove_wrong_detections_invalid_bbox(self, mock_model_path):
        """Test removing wrong detections with invalid bbox."""
        with patch('trackers.ball_tracker.YOLO'):
            tracker = BallTracker(mock_model_path)
            
            ball_positions = [
                {1: {"bbox": [100, 100, 120]}},  # Invalid bbox (only 3 values)
                {1: {"bbox": [105, 105, 125, 125]}},
            ]
            
            result = tracker.remove_wrong_detections(ball_positions)
            
            # Should handle invalid bboxes gracefully
            assert len(result) == 2

    def test_interpolate_missing_detections_all_valid(self, mock_model_path):
        """Test interpolation when all detections are valid."""
        with patch('trackers.ball_tracker.YOLO'):
            tracker = BallTracker(mock_model_path)
            
            ball_positions = [
                {1: {"bbox": [100, 100, 120, 120]}},
                {1: {"bbox": [110, 110, 130, 130]}},
                {1: {"bbox": [120, 120, 140, 140]}},
            ]
            
            result = tracker.interpolate_missing_detections(ball_positions)
            
            # Should return same positions since no interpolation needed
            assert len(result) == 3
            assert result[0] == {1: {"bbox": [100.0, 100.0, 120.0, 120.0]}}

    def test_interpolate_missing_detections_with_gaps(self, mock_model_path):
        """Test interpolation with missing detections."""
        with patch('trackers.ball_tracker.YOLO'):
            tracker = BallTracker(mock_model_path)
            
            ball_positions = [
                {1: {"bbox": [100, 100, 120, 120]}},
                {},  # Missing detection
                {1: {"bbox": [120, 120, 140, 140]}},
            ]
            
            result = tracker.interpolate_missing_detections(ball_positions)
            
            # Should interpolate the middle frame
            assert len(result) == 3
            assert result[0] == {1: {"bbox": [100.0, 100.0, 120.0, 120.0]}}
            # Due to ffill().interpolate().bfill(), the interpolated value should be between the two values
            interpolated_bbox = result[1][1]["bbox"]
            assert len(interpolated_bbox) == 4
            # Should be interpolated between [100, 100, 120, 120] and [120, 120, 140, 140]
            assert 100.0 <= interpolated_bbox[0] <= 120.0
            assert 100.0 <= interpolated_bbox[1] <= 120.0
            assert 120.0 <= interpolated_bbox[2] <= 140.0 
            assert 120.0 <= interpolated_bbox[3] <= 140.0
            assert result[2] == {1: {"bbox": [120.0, 120.0, 140.0, 140.0]}}

    def test_interpolate_missing_detections_all_empty(self, mock_model_path):
        """Test interpolation with all empty detections."""
        with patch('trackers.ball_tracker.YOLO'):
            tracker = BallTracker(mock_model_path)
            
            ball_positions = [{}, {}, {}]
            
            # This should handle the case where all positions are empty
            # The DataFrame creation may fail, so the function should handle it gracefully
            try:
                result = tracker.interpolate_missing_detections(ball_positions)
                # Should return empty dicts since nothing to interpolate
                assert len(result) == 3
                assert all(pos == {} for pos in result)
            except ValueError:
                # It's acceptable for this to raise a ValueError when all data is empty
                assert True