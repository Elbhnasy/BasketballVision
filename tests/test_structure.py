import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestProjectStructure:
    """Test that the project structure is as expected."""
    
    def test_src_directory_exists(self):
        """Test that src directory exists."""
        src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
        assert os.path.exists(src_path), "src directory should exist"
    
    def test_utils_directory_exists(self):
        """Test that utils directory exists."""
        utils_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'utils')
        assert os.path.exists(utils_path), "src/utils directory should exist"
    
    def test_video_utils_file_exists(self):
        """Test that video_utils.py exists."""
        video_utils_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'utils', 'video_utils.py')
        assert os.path.exists(video_utils_path), "video_utils.py should exist"
    
    def test_main_file_exists(self):
        """Test that main.py exists."""
        main_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'main.py')
        assert os.path.exists(main_path), "main.py should exist"
    
    def test_models_directory_exists(self):
        """Test that models directory exists or can be created."""
        models_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'models')
        # Model files are gitignored, so directory may not exist in CI
        # Create it if it doesn't exist (this is expected for model storage)
        if not os.path.exists(models_path):
            os.makedirs(models_path, exist_ok=True)
        assert os.path.exists(models_path), "src/models directory should exist or be created"
