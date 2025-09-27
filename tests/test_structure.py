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
        if not os.path.exists(models_path):
            os.makedirs(models_path, exist_ok=True)
        assert os.path.exists(models_path), "src/models directory should exist or be created"
    
    def test_trackers_directory_structure(self):
        """Test that trackers directory and files exist."""
        trackers_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'trackers')
        assert os.path.exists(trackers_path), "src/trackers directory should exist"
        
        # Check for tracker files
        ball_tracker_path = os.path.join(trackers_path, 'ball_tracker.py')
        player_tracker_path = os.path.join(trackers_path, 'player_tracker.py')
        init_path = os.path.join(trackers_path, '__init__.py')
        
        assert os.path.exists(ball_tracker_path), "ball_tracker.py should exist"
        assert os.path.exists(player_tracker_path), "player_tracker.py should exist"
        assert os.path.exists(init_path), "trackers/__init__.py should exist"
    
    def test_drawers_directory_structure(self):
        """Test that drawers directory and files exist."""
        drawers_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'drawers')
        assert os.path.exists(drawers_path), "src/drawers directory should exist"
        
        # Check for drawer files
        ball_drawer_path = os.path.join(drawers_path, 'ball_tracks_drawer.py')
        player_drawer_path = os.path.join(drawers_path, 'player_tracks_drawer.py')
        utils_path = os.path.join(drawers_path, 'utils.py')
        init_path = os.path.join(drawers_path, '__init__.py')
        
        assert os.path.exists(ball_drawer_path), "ball_tracks_drawer.py should exist"
        assert os.path.exists(player_drawer_path), "player_tracks_drawer.py should exist"
        assert os.path.exists(utils_path), "drawers/utils.py should exist"
        assert os.path.exists(init_path), "drawers/__init__.py should exist"
    
    def test_utils_files_exist(self):
        """Test that all utility files exist."""
        utils_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'utils')
        
        required_files = [
            '__init__.py',
            'video_utils.py',
            'bbox_utils.py', 
            'stubs_utils.py'
        ]
        
        for file_name in required_files:
            file_path = os.path.join(utils_path, file_name)
            assert os.path.exists(file_path), f"utils/{file_name} should exist"
    
    def test_videos_directory_structure(self):
        """Test that videos directory structure exists."""
        videos_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'videos')
        assert os.path.exists(videos_path), "src/videos directory should exist"
        
        input_path = os.path.join(videos_path, 'input')
        output_path = os.path.join(videos_path, 'output')
        
        assert os.path.exists(input_path), "src/videos/input directory should exist"
        assert os.path.exists(output_path), "src/videos/output directory should exist"
    
    def test_stubs_directory_exists(self):
        """Test that stubs directory exists."""
        stubs_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'stubs')
        assert os.path.exists(stubs_path), "src/stubs directory should exist"
    
    def test_tests_directory_structure(self):
        """Test that test directory is properly structured."""
        test_dir = os.path.dirname(__file__)
        
        # Check that test files exist
        required_test_files = [
            '__init__.py',
            'test_basic.py',
            'test_structure.py',
            'test_video_utils.py',
        ]
        
        for test_file in required_test_files:
            test_path = os.path.join(test_dir, test_file)
            assert os.path.exists(test_path), f"tests/{test_file} should exist"
    
    def test_project_root_files(self):
        """Test that project root contains required files."""
        project_root = os.path.join(os.path.dirname(__file__), '..')
        
        required_root_files = [
            'README.md',
            'LICENSE',
            'pyproject.toml',
            'uv.lock'
        ]
        
        for root_file in required_root_files:
            root_path = os.path.join(project_root, root_file)
            assert os.path.exists(root_path), f"{root_file} should exist in project root"
    
    def test_training_notebooks_exist(self):
        """Test that training notebooks directory exists."""
        notebooks_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'training_notebooks')
        assert os.path.exists(notebooks_path), "src/training_notebooks directory should exist"
