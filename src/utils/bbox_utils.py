def get_bbox_center(bbox: tuple) -> tuple:
    """
    Calculate the center of a bounding box.

    Args:
        bbox (tuple): A tuple containing (x1, y1, x2, y2).

    Returns:
        tuple: A tuple containing the center coordinates (x_center, y_center).
    """
    return (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2


def get_bbox_width(bbox: tuple) -> int:
    """
    Calculate the width of a bounding box.

    Args:
        bbox (tuple): A tuple containing (x1, y1, x2, y2).

    Returns:
        int: The width of the bounding box.
    """
    return bbox[2] - bbox[0]


def get_foot_position(bbox: tuple) -> tuple:
    """
    Calculate the foot position based on the bounding box.

    Args:
        bbox (tuple): A tuple containing (x1, y1, x2, y2).

    Returns:
        tuple: A tuple containing the foot position (x_foot, y_foot).
    """
    return (bbox[0], bbox[2] // 2), int(bbox[3])
