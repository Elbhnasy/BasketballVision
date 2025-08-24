import cv2
import numpy as np
from typing import Optional, Tuple
from utils.bbox_utils import get_bbox_center, get_bbox_width


def draw_triangle(
    frame: np.ndarray, bbox: Tuple[int, int, int, int], color: Tuple[int, int, int]
) -> np.ndarray:
    """
    Draw a triangle at the foot position of the player.

    Args:
        frame (np.ndarray): The video frame to draw on.
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).
        color (tuple): Color of the triangle in BGR format.
    """
    y = int(bbox[1])
    x, _ = get_bbox_center(bbox)

    triangle_points = np.array(
        [[x, y], [x - 10, y - 20], [x + 10, y - 20]], dtype=np.int32
    )
    cv2.fillPoly(frame, [triangle_points], color)
    cv2.polylines(frame, [triangle_points], isClosed=True, color=(0, 0, 0), thickness=2)

    return frame


def draw_ellipse(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    track_id: Optional[int],
) -> np.ndarray:
    """
    Draws an ellipse at the bottom of the bounding box (e.g., foot position),
    and optionally adds a label with track ID.

    Args:
        frame (np.ndarray): Input image/frame.
        bbox (Tuple[int, int, int, int]): Bounding box (x1, y1, x2, y2).
        color (Tuple[int, int, int]): Color in BGR format.
        track_id (Optional[int]): Track ID to display near the ellipse.

    Returns:
        np.ndarray: Frame with ellipse and optional ID drawn.
    """

    x_center, _ = get_bbox_center(bbox)
    width = get_bbox_width(bbox)
    y_bottom = int(bbox[3])

    axes = (int(width * 0.5), int(width * 0.175))

    cv2.ellipse(
        img=frame,
        center=(int(x_center), y_bottom),
        axes=axes,
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    if track_id is not None:
        rect_w, rect_h = 40, 20
        x1_rect = int(x_center) - rect_w // 2
        y1_rect = y_bottom + 15
        x2_rect = int(x_center) + rect_w // 2
        y2_rect = y_bottom + 15 + rect_h
        cv2.rectangle(
            frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color, thickness=cv2.FILLED
        )

        text = str(track_id)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = int(x_center) - text_size[0] // 2
        text_y = y1_rect + (rect_h + text_size[1]) // 2

        if track_id > 99:
            text_x -= 10

        cv2.putText(
            img=frame,
            text=text,
            org=(text_x, text_y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=(0, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    return frame
