import math
import carla
import py_trees
import shapely
import numpy as np

from .atomic.base import Criterion, to_numpy
from tools.timer import GameTime
from scenario_runner.ctn_operator import CtnSimOperator
    
class RunningRedLightTest(Criterion):
    """
    Check if an actor is running a red light

    Important parameters:
    - actor: CARLA actor to be used for this test
    - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
    """

    DISTANCE_LIGHT = 15  # m

    def __init__(
        self, 
        actor, 
        ctn_operator: CtnSimOperator,
        name="RunningRedLightTest", 
        terminate_on_failure=False
    ):
        """
        Init
        """
        super().__init__(name, actor, False, terminate_on_failure)

        self.ctn_operator = ctn_operator
        
        self._world = self.ctn_operator.get_world()
        self._map = self.ctn_operator.get_map()
        
        self._list_traffic_lights = []
        self._last_red_light_id = None
        self.debug = False

        all_actors = self._world.get_actors()
        for _actor in all_actors:  # pylint: disable=not-an-iterable
            if 'traffic_light' in _actor.type_id:
                center, waypoints = self.get_traffic_light_waypoints(_actor)
                self._list_traffic_lights.append((_actor, center, waypoints))
                
        self.st_detail = {
            "occurred": False,
            "details": []
        }

    # pylint: disable=no-self-use
    def is_vehicle_crossing_line(self, seg1, seg2):
        """
        check if vehicle crosses a line segment
        """
        line1 = shapely.geometry.LineString([(seg1[0].x, seg1[0].y), (seg1[1].x, seg1[1].y)])
        line2 = shapely.geometry.LineString([(seg2[0].x, seg2[0].y), (seg2[1].x, seg2[1].y)])
        inter = line1.intersection(line2)

        return not inter.is_empty

    def update(self):
        """
        Check if the actor is running a red light
        """
        new_status = py_trees.common.Status.RUNNING

        transform = self.actor.get_transform()
        location = transform.location
        if location is None:
            return new_status

        veh_extent = self.actor.bounding_box.extent.x

        tail_close_pt = self.rotate_point(carla.Vector3D(-0.8 * veh_extent, 0, 0), transform.rotation.yaw)
        tail_close_pt = location + carla.Location(tail_close_pt)

        tail_far_pt = self.rotate_point(carla.Vector3D(-veh_extent - 1, 0, 0), transform.rotation.yaw)
        tail_far_pt = location + carla.Location(tail_far_pt)

        for traffic_light, center, waypoints in self._list_traffic_lights:
            if self.debug:
                z = 2.1
                if traffic_light.state == carla.TrafficLightState.Red:
                    color = carla.Color(155, 0, 0)
                elif traffic_light.state == carla.TrafficLightState.Green:
                    color = carla.Color(0, 155, 0)
                else:
                    color = carla.Color(155, 155, 0)
                self._world.debug.draw_point(center + carla.Location(z=z), size=0.2, color=color, life_time=0.01)
                for wp in waypoints:
                    text = "{}.{}".format(wp.road_id, wp.lane_id)
                    self._world.debug.draw_string(
                        wp.transform.location + carla.Location(x=1, z=z), text, color=color, life_time=0.01
                    )
                    self._world.debug.draw_point(
                        wp.transform.location + carla.Location(z=z), size=0.1, color=color, life_time=0.01
                    )

            if self._last_red_light_id and self._last_red_light_id == traffic_light.id:
                continue
            
            center_loc = carla.Location(center)
            center_loc_np = to_numpy(center_loc)
            location_np = to_numpy(location)
            cl_dist = np.linalg.norm(center_loc_np - location_np)

            if cl_dist > self.DISTANCE_LIGHT:
                continue
            
            if traffic_light.state != carla.TrafficLightState.Red:
                continue

            for wp in waypoints:
                tail_wp = self._map.get_waypoint(tail_far_pt)

                # Calculate the dot product (Might be unscaled, as only its sign is important)
                actor_transform = self.actor.get_transform()
                ve_dir = to_numpy(actor_transform.get_forward_vector())
                wp_dir = to_numpy(wp.transform.get_forward_vector())

                # Check the lane until all the "tail" has passed
                
                if tail_wp.road_id == wp.road_id and tail_wp.lane_id == wp.lane_id and ve_dir.dot(wp_dir) > 0:
                    # This light is red and is affecting our lane
                    yaw_wp = wp.transform.rotation.yaw
                    lane_width = wp.lane_width
                    location_wp = wp.transform.location

                    lft_lane_wp = self.rotate_point(carla.Vector3D(0.6 * lane_width, 0, 0), yaw_wp + 90)
                    lft_lane_wp = location_wp + carla.Location(lft_lane_wp)
                    rgt_lane_wp = self.rotate_point(carla.Vector3D(0.6 * lane_width, 0, 0), yaw_wp - 90)
                    rgt_lane_wp = location_wp + carla.Location(rgt_lane_wp)

                    # Is the vehicle traversing the stop line?
                    if self.is_vehicle_crossing_line((tail_close_pt, tail_far_pt), (lft_lane_wp, rgt_lane_wp)) and traffic_light.state == carla.TrafficLightState.Red:
                        self.test_status = "FAILURE"
                        self.actual_value += 1
                        location = traffic_light.get_transform().location
                        
                        self.st_detail["occurred"] = True
                        self.st_detail["details"].append({
                            "id": self.actor.id,
                            "frame": GameTime.get_frame(),
                            "timestamp": GameTime.get_time(),
                            "other_id": traffic_light.id,
                            "location": {
                                "x": location.x,
                                "y": location.y,
                                "z": location.z
                            },
                            "state": str(traffic_light.state)
                        })
                        
                        self._last_red_light_id = traffic_light.id
                        break

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def rotate_point(self, point, angle):
        """
        rotate a given point by a given angle
        """
        x_ = math.cos(math.radians(angle)) * point.x - math.sin(math.radians(angle)) * point.y
        y_ = math.sin(math.radians(angle)) * point.x + math.cos(math.radians(angle)) * point.y
        return carla.Vector3D(x_, y_, point.z)

    def get_traffic_light_waypoints(self, traffic_light):
        """
        get area of a given traffic light
        """
        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)

        # Discretize the trigger box into points
        area_ext = traffic_light.trigger_volume.extent
        x_values = np.arange(-0.9 * area_ext.x, 0.9 * area_ext.x, 1.0)  # 0.9 to avoid crossing to adjacent lanes

        area = []
        for x in x_values:
            point = self.rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot)
            point_location = area_loc + carla.Location(x=point.x, y=point.y)
            area.append(point_location)

        # Get the waypoints of these points, removing duplicates
        ini_wps = []
        for pt in area:
            wpx = self._map.get_waypoint(pt)
            # As x_values are arranged in order, only the last one has to be checked
            if not ini_wps or ini_wps[-1].road_id != wpx.road_id or ini_wps[-1].lane_id != wpx.lane_id:
                ini_wps.append(wpx)

        # Advance them until the intersection
        wps = []
        for wpx in ini_wps:
            while not wpx.is_junction:
                next_wp = wpx.next(0.5)[0]
                if next_wp and not next_wp.is_junction:
                    wpx = next_wp
                else:
                    break
            wps.append(wpx)

        return area_loc, wps
