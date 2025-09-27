import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.bbox_utils import get_bbox_center, get_bbox_width, get_foot_position


class TestBboxUtils:
    """Test cases for bounding box utility functions."""
    
    def test_get_bbox_center_normal_case(self):
        """Test bbox center calculation with normal values."""
        bbox = (100, 50, 200, 150)
        center = get_bbox_center(bbox)
        
        expected_x = (100 + 200) // 2  # 150
        expected_y = (50 + 150) // 2   # 100
        
        assert center == (expected_x, expected_y)
        assert center == (150, 100)
    
    def test_get_bbox_center_zero_coordinates(self):
        """Test bbox center calculation with zero coordinates."""
        bbox = (0, 0, 100, 100)
        center = get_bbox_center(bbox)
        
        assert center == (50, 50)
    
    def test_get_bbox_center_negative_coordinates(self):
        """Test bbox center calculation with negative coordinates."""
        bbox = (-100, -50, 100, 50)
        center = get_bbox_center(bbox)
        
        expected_x = (-100 + 100) // 2  # 0
        expected_y = (-50 + 50) // 2    # 0
        
        assert center == (0, 0)
    
    def test_get_bbox_center_single_point(self):
        """Test bbox center calculation when bbox represents a single point."""
        bbox = (50, 50, 50, 50)
        center = get_bbox_center(bbox)
        
        assert center == (50, 50)
    
    def test_get_bbox_width_normal_case(self):
        """Test bbox width calculation with normal values."""
        bbox = (100, 50, 200, 150)
        width = get_bbox_width(bbox)
        
        expected_width = 200 - 100  # 100
        assert width == 100
    
    def test_get_bbox_width_zero_width(self):
        """Test bbox width calculation when width is zero."""
        bbox = (100, 50, 100, 150)
        width = get_bbox_width(bbox)
        
        assert width == 0
    
    def test_get_bbox_width_negative_result(self):
        """Test bbox width calculation with x2 < x1 (edge case)."""
        bbox = (200, 50, 100, 150)  # x2 < x1
        width = get_bbox_width(bbox)
        
        assert width == -100  # This might indicate an invalid bbox
    
    def test_get_bbox_width_large_values(self):
        """Test bbox width calculation with large coordinate values."""
        bbox = (1000, 500, 2000, 1500)
        width = get_bbox_width(bbox)
        
        assert width == 1000
    
    def test_get_foot_position_normal_case(self):
        """Test foot position calculation with normal values."""
        bbox = (100, 50, 200, 150)
        foot_pos = get_foot_position(bbox)
        
        # Expected: ((x1, x2//2), y2) = ((100, 200//2), 150) = ((100, 100), 150)
        expected_x_tuple = (100, 200 // 2)  # (100, 100)
        expected_y = 150
        
        assert foot_pos == (expected_x_tuple, expected_y)
        assert foot_pos == ((100, 100), 150)
    
    def test_get_foot_position_zero_coordinates(self):
        """Test foot position calculation with zero coordinates."""
        bbox = (0, 0, 100, 100)
        foot_pos = get_foot_position(bbox)
        
        expected_x_tuple = (0, 100 // 2)  # (0, 50)
        expected_y = 100
        
        assert foot_pos == ((0, 50), 100)
    
    def test_get_foot_position_odd_x2_coordinate(self):
        """Test foot position calculation with odd x2 coordinate."""
        bbox = (10, 20, 101, 150)  # x2 = 101 (odd number)
        foot_pos = get_foot_position(bbox)
        
        expected_x_tuple = (10, 101 // 2)  # (10, 50)
        expected_y = 150
        
        assert foot_pos == ((10, 50), 150)
    
    def test_get_foot_position_negative_coordinates(self):
        """Test foot position calculation with negative coordinates."""
        bbox = (-100, -50, 100, 50)
        foot_pos = get_foot_position(bbox)
        
        expected_x_tuple = (-100, 100 // 2)  # (-100, 50)
        expected_y = 50
        
        assert foot_pos == ((-100, 50), 50)


class TestBboxUtilsEdgeCases:
    """Test edge cases and error conditions for bbox utils."""
    
    def test_all_functions_with_float_coordinates(self):
        """Test all functions with float coordinates (should work due to Python's flexibility)."""
        bbox = (100.5, 50.5, 200.5, 150.5)
        
        # get_bbox_center should handle floats but return integers due to //
        center = get_bbox_center(bbox)
        assert center == (150, 100)  # (301.0 // 2, 201.0 // 2) = (150, 100)
        
        # get_bbox_width should work with floats
        width = get_bbox_width(bbox)
        assert width == 100.0
        
        # get_foot_position should work with floats
        foot_pos = get_foot_position(bbox)
        assert foot_pos == ((100.5, 100.0), 150)  # ((100.5, 200.5//2), int(150.5))
    
    def test_bbox_with_very_large_numbers(self):
        """Test functions with very large coordinate values."""
        bbox = (1000000, 2000000, 3000000, 4000000)
        
        center = get_bbox_center(bbox)
        assert center == (2000000, 3000000)
        
        width = get_bbox_width(bbox)
        assert width == 2000000
        
        foot_pos = get_foot_position(bbox)
        assert foot_pos == ((1000000, 1500000), 4000000)
    
    def test_minimum_valid_bbox(self):
        """Test functions with minimum valid bbox values."""
        bbox = (0, 0, 1, 1)
        
        center = get_bbox_center(bbox)
        assert center == (0, 0)  # (1 // 2, 1 // 2) = (0, 0)
        
        width = get_bbox_width(bbox)
        assert width == 1
        
        foot_pos = get_foot_position(bbox)
        assert foot_pos == ((0, 0), 1)  # ((0, 1//2), 1) = ((0, 0), 1)