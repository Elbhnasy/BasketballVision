import pytest
import sys
import os
import tempfile
import shutil
from unittest.mock import patch, Mock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.stubs_utils import save_stub, read_stub


class TestStubsUtils:
    """Test cases for stub utility functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Provide sample data for testing."""
        return {
            "tracks": [
                {1: {"bbox": [100, 100, 200, 200]}},
                {1: {"bbox": [110, 110, 210, 210]}, 2: {"bbox": [300, 300, 400, 400]}}
            ],
            "metadata": {"fps": 30, "total_frames": 2}
        }
    
    def test_save_stub_success(self, tmp_path, sample_data):
        """Test successful saving of stub data."""
        stub_path = tmp_path / "test_stub.pkl"
        
        with patch('torch.save') as mock_torch_save:
            save_stub(str(stub_path), sample_data)
            
            # Verify torch.save was called with correct arguments
            mock_torch_save.assert_called_once_with(sample_data, str(stub_path))
    
    def test_save_stub_creates_directory(self, tmp_path, sample_data):
        """Test that save_stub creates necessary directories."""
        # Create a nested path that doesn't exist
        nested_path = tmp_path / "nested" / "deep" / "test_stub.pkl"
        
        with patch('torch.save') as mock_torch_save:
            save_stub(str(nested_path), sample_data)
            
            # Directory should be created
            assert nested_path.parent.exists()
            mock_torch_save.assert_called_once_with(sample_data, str(nested_path))
    
    def test_save_stub_torch_save_failure(self, tmp_path, sample_data):
        """Test save_stub behavior when torch.save fails."""
        stub_path = tmp_path / "test_stub.pkl"
        
        with patch('torch.save') as mock_torch_save:
            mock_torch_save.side_effect = Exception("Disk full")
            
            # Should not raise exception, just log error
            save_stub(str(stub_path), sample_data)
            
            mock_torch_save.assert_called_once_with(sample_data, str(stub_path))
    
    def test_save_stub_with_different_data_types(self, tmp_path):
        """Test saving different types of data."""
        
        test_cases = [
            [],  # Empty list
            {},  # Empty dict
            [1, 2, 3, 4, 5],  # Simple list
            {"key": "value", "number": 42},  # Simple dict
            None,  # None value
            "string_data",  # String
        ]
        
        for i, data in enumerate(test_cases):
            test_path = tmp_path / f"test_stub_{i}.pkl"
            
            with patch('torch.save') as mock_torch_save:
                save_stub(str(test_path), data)
                mock_torch_save.assert_called_once_with(data, str(test_path))
    
    def test_read_stub_skip_reading(self, tmp_path):
        """Test read_stub when read_from_stub is False."""
        stub_path = tmp_path / "test_stub.pkl"
        
        result = read_stub(False, str(stub_path))
        
        assert result is None
    
    def test_read_stub_file_not_exists(self, tmp_path):
        """Test read_stub when stub file doesn't exist."""
        stub_path = tmp_path / "non_existent.pkl"
        
        result = read_stub(True, str(stub_path))
        
        assert result is None
    
    def test_read_stub_success(self, tmp_path, sample_data):
        """Test successful reading of stub data."""
        stub_path = tmp_path / "test_stub.pkl"
        
        # Create an empty file to simulate existing file
        stub_path.write_text("dummy")
        
        with patch('torch.load') as mock_torch_load:
            mock_torch_load.return_value = sample_data
            
            result = read_stub(True, str(stub_path), map_location="cuda")
            
            assert result == sample_data
            mock_torch_load.assert_called_once_with(str(stub_path), map_location="cuda")
    
    def test_read_stub_torch_load_failure(self, tmp_path):
        """Test read_stub behavior when torch.load fails."""
        stub_path = tmp_path / "test_stub.pkl"
        
        # Create an empty file to simulate existing file
        stub_path.write_text("dummy")
        
        with patch('torch.load') as mock_torch_load:
            mock_torch_load.side_effect = Exception("Corrupted file")
            
            result = read_stub(True, str(stub_path))
            
            assert result is None
            mock_torch_load.assert_called_once_with(str(stub_path), map_location="cpu")
    
    def test_read_stub_default_map_location(self, tmp_path, sample_data):
        """Test read_stub with default map_location parameter."""
        stub_path = tmp_path / "test_stub.pkl"
        
        # Create an empty file to simulate existing file
        stub_path.write_text("dummy")
        
        with patch('torch.load') as mock_torch_load:
            mock_torch_load.return_value = sample_data
            
            result = read_stub(True, str(stub_path))  # No map_location specified
            
            assert result == sample_data
            mock_torch_load.assert_called_once_with(str(stub_path), map_location="cpu")
    
    def test_read_stub_different_map_locations(self, tmp_path, sample_data):
        """Test read_stub with different map_location values."""
        
        map_locations = ["cpu", "cuda", "cuda:0", "cuda:1"]
        
        for map_location in map_locations:
            stub_path = tmp_path / f"test_stub_{map_location.replace(':', '_')}.pkl"
            
            # Create an empty file to simulate existing file for each test
            stub_path.write_text("dummy")
                
            with patch('torch.load') as mock_torch_load:
                mock_torch_load.return_value = sample_data
                
                result = read_stub(True, str(stub_path), map_location=map_location)
                
                assert result == sample_data
                mock_torch_load.assert_called_once_with(str(stub_path), map_location=map_location)


class TestStubsUtilsIntegration:
    """Integration tests for save and read operations together."""
    
    @pytest.fixture
    def sample_data(self):
        """Provide sample data for integration testing."""
        return {
            "tracks": [
                {1: {"bbox": [100, 100, 200, 200]}},
                {1: {"bbox": [110, 110, 210, 210]}, 2: {"bbox": [300, 300, 400, 400]}}
            ],
            "metadata": {"fps": 30, "total_frames": 2}
        }
    
    def test_save_and_read_roundtrip(self, tmp_path, sample_data):
        """Test complete save and read cycle with real torch operations."""
        stub_path = tmp_path / "roundtrip_test.pkl"
        
        # Save the data
        save_stub(str(stub_path), sample_data)
        
        # Verify file was created
        assert stub_path.exists()
        
        # Read the data back
        result = read_stub(True, str(stub_path))
        
        # Should get back the same data
        assert result == sample_data
    
    def test_save_and_read_empty_data(self, tmp_path):
        """Test roundtrip with empty data structures."""
        stub_path = tmp_path / "empty_test.pkl"
        
        test_cases = [[], {}, None]
        
        for empty_data in test_cases:
            save_stub(str(stub_path), empty_data)
            result = read_stub(True, str(stub_path))
            assert result == empty_data
    
    def test_multiple_saves_same_path(self, tmp_path):
        """Test multiple saves to the same path (should overwrite)."""
        stub_path = tmp_path / "overwrite_test.pkl"
        
        # Save first data
        first_data = {"version": 1, "data": [1, 2, 3]}
        save_stub(str(stub_path), first_data)
        
        # Save second data (should overwrite)
        second_data = {"version": 2, "data": [4, 5, 6]}
        save_stub(str(stub_path), second_data)
        
        # Read back should get the second data
        result = read_stub(True, str(stub_path))
        assert result == second_data
        assert result != first_data
    
    def test_concurrent_access_simulation(self, tmp_path, sample_data):
        """Test behavior when multiple operations try to access the same file."""
        stub_path = tmp_path / "concurrent_test.pkl"
        
        # Save data first
        save_stub(str(stub_path), sample_data)
        
        # Simulate multiple read attempts
        for i in range(5):
            result = read_stub(True, str(stub_path))
            assert result == sample_data
    
    def test_directory_creation_edge_cases(self, tmp_path):
        """Test directory creation with various path scenarios."""
        test_cases = [
            tmp_path / "single" / "level.pkl",
            tmp_path / "deep" / "nested" / "multiple" / "levels.pkl",
            tmp_path / "with spaces" / "in path" / "file.pkl",
        ]
        
        for stub_path in test_cases:
            save_stub(str(stub_path), {"test": "data"})
            
            # Verify directory was created
            assert stub_path.parent.exists()
            
            # Verify we can read back the data
            result = read_stub(True, str(stub_path))
            assert result == {"test": "data"}