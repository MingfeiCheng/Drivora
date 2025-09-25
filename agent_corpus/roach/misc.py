import math
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import List, Dict
from shapely.geometry import Point, Polygon, LineString

### use for offline analysis
@dataclass
class Vector3D:
    x: float
    y: float
    z: float

    def distance(self, target):
        self_point = Point(self.x, self.y)
        target_point = Point(target.x, target.y)
        return self_point.distance(target_point)

@dataclass
class Rotation3D:
    pitch: float
    yaw: float
    roll: float
#############################

def lateral_shift(transform, shift):
    transform.rotation.yaw += 90
    shifted = transform.location + shift * transform.get_forward_vector()
    return [shifted.x, shifted.y]

def rotate_point(point, theta) -> Point:
    radians = theta / 180 * np.pi
    x = point.x
    y = point.y

    rotated_x = x * math.cos(radians) - y * math.sin(radians)
    rotated_y = x * math.sin(radians) + y * math.cos(radians)

    return Point(rotated_x, rotated_y)


def inverse_rotate_point(point, theta) -> Point:
    radians = theta / 180 * np.pi
    x = point.x
    y = point.y

    rotated_x = x * math.cos(radians) + y * math.sin(radians)
    rotated_y = x * math.sin(radians) - y * math.cos(radians)

    return Point(rotated_x, rotated_y)


def get_local_bounding_box_vertices(length, width) -> List[Point]:
    cx = 0.0
    cy = 0.0
    half_width = width / 2
    half_length = length / 2

    # corners of the bounding box before rotation
    corners = [
        (cx - half_length, cy - half_width), #
        (cx + half_length, cy - half_width),
        (cx + half_length, cy + half_width),
        (cx - half_length, cy + half_width)
    ]

    # rotate each corner
    rotated_corners = [Point(corner) for corner in corners]
    return rotated_corners

def rotate_point_traffic_signal(point, radians):
    """
    rotate a given point by a given angle
    """
    rotated_x = math.cos(radians) * point.x - math.sin(radians) * point.y
    rotated_y = math.sin(radians) * point.x - math.cos(radians) * point.y

    return Vector3D(rotated_x, rotated_y, point.z)
### for traffic light & traffic sign


### visualization
def draw_scene_object_patch(category: str, polygon: Polygon):
    category_color = {
        'ego': '#f54545',
        'npc_vehicle': '#3a74fc',
        'npc_walker': '#fc9e3a',
        'npc_static': '#a55bf5'
    }

    polygon = patches.Polygon(list(polygon.exterior.coords),
                              closed=True,
                              color=category_color[category],
                              alpha=0.8)


    return polygon

def vis_scene(ego_polygon: Polygon,
              ego_plan: List[Point],
              npc_vehicle_polygons: List,
              npc_walker_polygons: List[Polygon],
              npc_static_polygons: List[Polygon],
              road_boundaries: List[LineString],
              traffic_vectors: List[LineString],
              # traffic_light: str,
              # traffic_sign: str,
              ego_velocity: str,
              ego_acceleration: str,
              ego_control: str,
              save_scene_file: str,
              ax_size: float = 30.0):

    fig, ax = plt.subplots()

    # draw ego
    ego_patch = draw_scene_object_patch('ego', ego_polygon)
    ax.add_patch(ego_patch)

    ax.scatter(x=ego_plan[0], y=ego_plan[1], s=3, c='#a60505', alpha=0.8)

    for polygon in npc_vehicle_polygons:
        # print(polygon)
        polygon_patch = draw_scene_object_patch('npc_vehicle', polygon)
        ax.add_patch(polygon_patch)

    for polygon in npc_walker_polygons:
        polygon_patch = draw_scene_object_patch('npc_walker', polygon)
        ax.add_patch(polygon_patch)

    for polygon in npc_static_polygons:
        polygon_patch = draw_scene_object_patch('npc_static', polygon)
        ax.add_patch(polygon_patch)

    for road_linestring in road_boundaries:
        x, y = road_linestring.xy
        ax.plot(x, y, 'black')

    # draw vector #99e374
    for tv in traffic_vectors:
        x, y = tv.xy
        ax.plot(x, y, '#99e374', zorder=0, alpha=0.7)

    caption = f"{ego_velocity} {ego_acceleration}\n{ego_control}"

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-ax_size, ax_size)
    ax.set_ylim(-ax_size, ax_size)
    fig.subplots_adjust(bottom=0.2)
    plt.figtext(0.5, 0.03, caption, ha="center", fontsize=10, color="black", wrap=False)
    plt.show()
    plt.savefig(save_scene_file)
    plt.close("all")

def get_forward_value(transform, velocity):
    """ Convert the vehicle transform directly to forward speed """
    vel_np = np.array([velocity.x, velocity.y, velocity.z])
    pitch = np.deg2rad(transform.rotation.pitch)
    yaw = np.deg2rad(transform.rotation.yaw)
    orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    speed = np.dot(vel_np, orientation)
    return speed

def vis_scene_np(scene_np: Dict, save_scene_file, x_min: float = -10.0, x_max: float = 20.0, y_min: float = -20.0, y_max: float = 20.0, retrieval_dist: float = -1.0):

    ego_polygon = scene_np['ego_polygon']
    ego_plan = scene_np['ego_plan']
    ego_velocity = scene_np['ego_velocity']
    ego_acceleration = scene_np['ego_acceleration']
    ego_road = scene_np['ego_road']
    ego_control = scene_np['ego_control']
    npc_vehicle_polygons = scene_np['npc_vehicle_polygons']
    npc_walker_polygons = scene_np['npc_walker_polygons']
    npc_static_polygons = scene_np['npc_static_polygons']

    fig, ax = plt.subplots()

    # draw ego
    ego_patch = draw_scene_object_patch('ego', ego_polygon)
    ax.add_patch(ego_patch)

    ax.scatter(x=ego_plan[0], y=ego_plan[1], s=3, c='#a60505', alpha=0.8)

    for polygon in npc_vehicle_polygons:
        # print(polygon)
        polygon_patch = draw_scene_object_patch('npc_vehicle', polygon)
        ax.add_patch(polygon_patch)

    for polygon in npc_walker_polygons:
        polygon_patch = draw_scene_object_patch('npc_walker', polygon)
        ax.add_patch(polygon_patch)

    for polygon in npc_static_polygons:
        polygon_patch = draw_scene_object_patch('npc_static', polygon)
        ax.add_patch(polygon_patch)

    caption = f"{ego_velocity} {ego_acceleration}\n{ego_control}\n{ego_road}\ndist: {retrieval_dist}"

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    fig.subplots_adjust(bottom=0.2)
    plt.figtext(0.5, 0.03, caption, ha="center", fontsize=10, color="black", wrap=False)
    plt.show()
    plt.savefig(save_scene_file)
    plt.close("all")