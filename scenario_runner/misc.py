import math

from typing import List, Tuple

def calculate_bbox_polygon_2d(
    actor_location_x: float,
    actor_location_y: float,
    actor_yaw: float,
    bbox_extent_x: float,
    bbox_extent_y: float,
    bbox_rotation_yaw: float
) -> List[Tuple[float, float]]:

    yaw = math.radians(actor_yaw + bbox_rotation_yaw)

    local_corners = [
        ( bbox_extent_x,  bbox_extent_y),
        ( bbox_extent_x, -bbox_extent_y),
        (-bbox_extent_x, -bbox_extent_y),
        (-bbox_extent_x,  bbox_extent_y),
    ]

    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)

    world_corners = []
    for x, y in local_corners:
        wx = actor_location_x + (x * cos_yaw - y * sin_yaw)
        wy = actor_location_y + (x * sin_yaw + y * cos_yaw)
        world_corners.append([wx, wy])

    return world_corners