import pytest
import sys
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trackers.player_tracker import PlayerTracker


class TestPlayerTracker:
    """Comprehensive tests for PlayerTracker class."""
    
    @pytest.fixture
    def mock_model_path(self, tmp_path):
        """Create a temporary model file."""
        model_file = tmp_path / "test_player_model.pt"
        model_file.touch()
        return str(model_file)
    
    @pytest.fixture
    def mock_yolo_model(self):
        """Create a mock YOLO model."""
        mock_model = Mock()
        mock_model.names = {0: "Player"}
        mock_model.to = Mock()
        return mock_model
    
    @pytest.fixture
    def sample_frames(self):
        """Generate sample frames for testing."""
        return [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)]
    
    @pytest.fixture
    def mock_byte_track(self):
        """Create a mock ByteTrack tracker."""
        mock_tracker = Mock()
        return mock_tracker

    def test_init_with_valid_model_path(self, mock_model_path):
        """Test initialization with valid model path."""
        with patch('trackers.player_tracker.YOLO') as mock_yolo, \
             patch('trackers.player_tracker.sv.ByteTrack') as mock_bytetrack:
            
            mock_model = Mock()
            mock_model.to = Mock()
            mock_yolo.return_value = mock_model
            
            tracker = PlayerTracker(mock_model_path, batch_size=15, device="cuda")
            
            assert tracker.batch_size == 15
            assert tracker.device == "cuda"
            assert tracker.model == mock_model
            mock_yolo.assert_called_once_with(mock_model_path)
            mock_model.to.assert_called_once_with("cuda")
            mock_bytetrack.assert_called_once()

    def test_init_with_invalid_model_path(self):
        """Test initialization with invalid model path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            PlayerTracker("non_existent_player_model.pt")

    def test_init_default_parameters(self, mock_model_path):
        """Test initialization with default parameters."""
        with patch('trackers.player_tracker.YOLO') as mock_yolo, \
             patch('trackers.player_tracker.sv.ByteTrack') as mock_bytetrack:
            
            mock_model = Mock()
            mock_model.to = Mock()
            mock_yolo.return_value = mock_model
            
            tracker = PlayerTracker(mock_model_path)
            
            assert tracker.batch_size == 20
            assert tracker.device == "cpu"

    @patch('trackers.player_tracker.sv.ByteTrack')
    @patch('trackers.player_tracker.YOLO')
    def test_detect_frames_single_batch(self, mock_yolo, mock_bytetrack, mock_model_path, sample_frames):
        """Test detection with frames that fit in a single batch."""
        mock_model = Mock()
        mock_model.to = Mock()
        mock_yolo.return_value = mock_model
        
        tracker = PlayerTracker(mock_model_path, batch_size=10)
        
        # Mock the model call to return detection results
        mock_detections = [Mock() for _ in range(5)]
        mock_model.return_value = mock_detections
        mock_model.side_effect = lambda frames, conf, verbose: mock_detections[:len(frames)]
        
        result = tracker.detect_frames(sample_frames, conf=0.7)
        
        assert len(result) == 5
        mock_model.assert_called_once_with(sample_frames, conf=0.7, verbose=False)

    @patch('trackers.player_tracker.sv.ByteTrack')
    @patch('trackers.player_tracker.YOLO')
    def test_detect_frames_multiple_batches(self, mock_yolo, mock_bytetrack, mock_model_path):
        """Test detection with frames requiring multiple batches."""
        mock_model = Mock()
        mock_model.to = Mock()
        mock_yolo.return_value = mock_model
        
        # Create 35 frames with batch_size=10
        frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(35)]
        tracker = PlayerTracker(mock_model_path, batch_size=10)
        
        # Mock the model to return different results for each batch
        mock_model.side_effect = lambda batch, conf, verbose: [Mock() for _ in batch]
        
        result = tracker.detect_frames(frames, conf=0.5)
        
        assert len(result) == 35
        assert mock_model.call_count == 4  # 35 frames / 10 batch_size = 4 calls

    @patch('trackers.player_tracker.save_stub')
    @patch('trackers.player_tracker.read_stub')
    @patch('trackers.player_tracker.sv.ByteTrack')
    @patch('trackers.player_tracker.YOLO')
    def test_get_object_tracks_with_cached_data(self, mock_yolo, mock_bytetrack, mock_read_stub, mock_save_stub, mock_model_path, sample_frames):
        """Test getting tracks when cached data is available."""
        mock_model = Mock()
        mock_model.to = Mock()
        mock_yolo.return_value = mock_model
        
        # Mock cached tracks
        cached_tracks = [{1: {"bbox": [100, 100, 200, 200]}, 2: {"bbox": [300, 300, 400, 400]}} for _ in range(5)]
        mock_read_stub.return_value = cached_tracks
        
        tracker = PlayerTracker(mock_model_path)
        result = tracker.get_object_tracks(sample_frames, read_from_stub=True, stub_path="player_test.pkl")
        
        assert result == cached_tracks
        mock_read_stub.assert_called_once_with(True, "player_test.pkl")
        mock_save_stub.assert_not_called()

    @patch('trackers.player_tracker.save_stub')
    @patch('trackers.player_tracker.read_stub')
    @patch('trackers.player_tracker.sv.Detections')
    @patch('trackers.player_tracker.sv.ByteTrack')
    @patch('trackers.player_tracker.YOLO')
    def test_get_object_tracks_no_cached_data(self, mock_yolo, mock_bytetrack, mock_sv_detections, mock_read_stub, mock_save_stub, mock_model_path, sample_frames):
        """Test getting tracks when no cached data is available."""
        mock_model = Mock()
        mock_model.to = Mock()
        mock_yolo.return_value = mock_model
        
        # Mock ByteTrack tracker
        mock_tracker = Mock()
        mock_bytetrack.return_value = mock_tracker
        
        # Mock no cached data
        mock_read_stub.return_value = None
        
        # Mock detection result - one for each frame in sample_frames (5 frames)
        mock_detection = Mock()
        mock_detection.names = {0: "Player"}
        mock_model.return_value = [mock_detection] * 5  # One detection per frame
        
        # Mock supervision detections
        mock_sv_detection = Mock()
        mock_sv_detection.class_id = np.array([0])
        mock_sv_detection.__getitem__ = Mock(return_value=mock_sv_detection)  # For filtering
        mock_sv_detections.from_ultralytics.return_value = mock_sv_detection
        
        # Mock tracked detections
        mock_tracked = Mock()
        mock_tracked.tracker_id = np.array([1, 2])
        mock_tracked.xyxy = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
        mock_tracker.update_with_detections.return_value = mock_tracked
        
        tracker = PlayerTracker(mock_model_path)
        result = tracker.get_object_tracks(sample_frames, stub_path="player_test.pkl")

        # Should have results for the number of frames provided (5 frames in sample_frames)
        assert len(result) == len(sample_frames)
        assert len(result) == 5
        mock_save_stub.assert_called_once_with("player_test.pkl", result)

    @patch('trackers.player_tracker.sv.ByteTrack')
    @patch('trackers.player_tracker.YOLO')
    def test_get_object_tracks_empty_frames(self, mock_yolo, mock_bytetrack, mock_model_path):
        """Test getting tracks with empty frame list."""
        mock_model = Mock()
        mock_model.to = Mock()
        mock_yolo.return_value = mock_model
        
        tracker = PlayerTracker(mock_model_path)
        result = tracker.get_object_tracks([])
        
        assert result == []

    @patch('trackers.player_tracker.save_stub')
    @patch('trackers.player_tracker.read_stub')
    @patch('trackers.player_tracker.sv.Detections')
    @patch('trackers.player_tracker.sv.ByteTrack')
    @patch('trackers.player_tracker.YOLO')
    def test_get_object_tracks_no_player_class(self, mock_yolo, mock_bytetrack, mock_sv_detections, mock_read_stub, mock_save_stub, mock_model_path, sample_frames):
        """Test getting tracks when model doesn't have Player class."""
        mock_model = Mock()
        mock_model.to = Mock()
        mock_yolo.return_value = mock_model
        
        mock_tracker = Mock()
        mock_bytetrack.return_value = mock_tracker
        mock_read_stub.return_value = None
        
        # Mock detection result without Player class - one for each frame
        mock_detection = Mock()
        mock_detection.names = {0: "Ball", 1: "Car"}
        mock_model.return_value = [mock_detection] * 5  # One detection per frame
        
        # Mock supervision detections
        mock_sv_detection = Mock()
        mock_sv_detections.from_ultralytics.return_value = mock_sv_detection
        
        # Mock tracked detections with no tracker_id (empty tracking result)
        mock_tracked = Mock()
        mock_tracked.tracker_id = None
        mock_tracker.update_with_detections.return_value = mock_tracked
        
        tracker = PlayerTracker(mock_model_path)
        result = tracker.get_object_tracks(sample_frames)
        
        # Should return empty dicts for each frame (5 frames in sample_frames)
        assert len(result) == len(sample_frames)
        assert len(result) == 5
        assert all(track == {} for track in result)

    @patch('trackers.player_tracker.save_stub')
    @patch('trackers.player_tracker.read_stub')
    @patch('trackers.player_tracker.sv.Detections')
    @patch('trackers.player_tracker.sv.ByteTrack')
    @patch('trackers.player_tracker.YOLO')
    def test_get_object_tracks_no_detections(self, mock_yolo, mock_bytetrack, mock_sv_detections, mock_read_stub, mock_save_stub, mock_model_path, sample_frames):
        """Test getting tracks when no players are detected."""
        mock_model = Mock()
        mock_model.to = Mock()
        mock_yolo.return_value = mock_model
        
        mock_tracker = Mock()
        mock_bytetrack.return_value = mock_tracker
        mock_read_stub.return_value = None
        
        # Mock detection result with Player class - one for each frame
        mock_detection = Mock()
        mock_detection.names = {0: "Player"}
        mock_model.return_value = [mock_detection] * 5  # One detection per frame
        
        # Mock supervision detections with Player class but no actual detections
        mock_sv_detection = Mock()
        mock_sv_detection.class_id = np.array([])  # No detections
        mock_sv_detection.__getitem__ = Mock(return_value=mock_sv_detection)
        mock_sv_detections.from_ultralytics.return_value = mock_sv_detection
        
        # Mock tracked detections with no tracker_id
        mock_tracked = Mock()
        mock_tracked.tracker_id = None
        mock_tracker.update_with_detections.return_value = mock_tracked
        
        tracker = PlayerTracker(mock_model_path)
        result = tracker.get_object_tracks(sample_frames)
        
        # Should return empty dicts for each frame (5 frames in sample_frames)
        assert len(result) == len(sample_frames)
        assert len(result) == 5
        assert all(track == {} for track in result)

    @patch('trackers.player_tracker.save_stub')
    @patch('trackers.player_tracker.read_stub')
    @patch('trackers.player_tracker.sv.Detections')
    @patch('trackers.player_tracker.sv.ByteTrack')
    @patch('trackers.player_tracker.YOLO')
    def test_get_object_tracks_multiple_players(self, mock_yolo, mock_bytetrack, mock_sv_detections, mock_read_stub, mock_save_stub, mock_model_path, sample_frames):
        """Test getting tracks with multiple players detected."""
        mock_model = Mock()
        mock_model.to = Mock()
        mock_yolo.return_value = mock_model
        
        mock_tracker = Mock()
        mock_bytetrack.return_value = mock_tracker
        mock_read_stub.return_value = None
        
        # Mock detection result with Player class
        mock_detection = Mock()
        mock_detection.names = {0: "Player"}
        mock_model.return_value = [mock_detection]
        
        # Mock supervision detections
        mock_sv_detection = Mock()
        mock_sv_detection.class_id = np.array([0, 0, 0])  # 3 player detections
        mock_sv_detection.__getitem__ = Mock(return_value=mock_sv_detection)
        mock_sv_detections.from_ultralytics.return_value = mock_sv_detection
        
        # Mock tracked detections with multiple tracker IDs
        mock_tracked = Mock()
        mock_tracked.tracker_id = np.array([1, 2, 3])
        mock_tracked.xyxy = np.array([[100, 100, 200, 200], [300, 300, 400, 400], [500, 500, 600, 600]])
        mock_tracker.update_with_detections.return_value = mock_tracked
        
        tracker = PlayerTracker(mock_model_path)
        result = tracker.get_object_tracks(sample_frames[:1])  # Test with one frame
        
        # Should contain tracks for 3 players
        assert len(result) == 1
        assert len(result[0]) == 3  # 3 tracked players
        assert 1 in result[0]
        assert 2 in result[0]
        assert 3 in result[0]

    @patch('trackers.player_tracker.save_stub')
    @patch('trackers.player_tracker.read_stub')
    def test_get_object_tracks_cached_data_length_mismatch(self, mock_read_stub, mock_save_stub, mock_model_path, sample_frames):
        """Test behavior when cached data length doesn't match frames."""
        with patch('trackers.player_tracker.YOLO') as mock_yolo, \
             patch('trackers.player_tracker.sv.ByteTrack') as mock_bytetrack, \
             patch('trackers.player_tracker.sv.Detections') as mock_sv_detections:
            
            mock_model = Mock()
            mock_model.to = Mock()
            mock_yolo.return_value = mock_model
            
            # Mock ByteTrack tracker
            mock_tracker = Mock()
            mock_bytetrack.return_value = mock_tracker
            
            # Mock cached data with different length
            cached_tracks = [{1: {"bbox": [100, 100, 200, 200]}}]  # Only 1 track vs sample_frames length
            mock_read_stub.return_value = cached_tracks
            
            # Mock save_stub to prevent filesystem errors
            mock_save_stub.return_value = None
            
            tracker = PlayerTracker(mock_model_path)
            
            # Mock the detection pipeline for when cache is invalid
            mock_detection = Mock()
            mock_detection.names = {0: "Player"}
            mock_model.return_value = [mock_detection] * 5  # One detection per frame
            
            # Mock supervision detections
            mock_sv_detection = Mock()
            mock_sv_detection.class_id = np.array([0])
            mock_sv_detection.__getitem__ = Mock(return_value=mock_sv_detection)
            mock_sv_detections.from_ultralytics.return_value = mock_sv_detection
            
            # Mock tracked detections
            mock_tracked = Mock()
            mock_tracked.tracker_id = np.array([1])
            mock_tracked.xyxy = np.array([[100, 100, 200, 200]])
            mock_tracker.update_with_detections.return_value = mock_tracked
            
            result = tracker.get_object_tracks(sample_frames, read_from_stub=True, stub_path="test.pkl")
            
            # Should proceed with detection since cached data length doesn't match
            # Result length should match sample_frames length (5 frames)
            assert len(result) == len(sample_frames)
            assert len(result) == 5