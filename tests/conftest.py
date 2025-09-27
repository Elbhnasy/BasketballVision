# Test Configuration and Fixtures
# This file contains common test configurations and fixtures

import pytest
import sys
import os
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# Common fixtures that can be used across test files
@pytest.fixture
def sample_frame():
    """Generate a sample video frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture  
def sample_bbox():
    """Generate a sample bounding box."""
    return [100, 100, 200, 200]


@pytest.fixture
def sample_track_data():
    """Generate sample tracking data."""
    return {1: {"bbox": [100, 100, 200, 200]}}