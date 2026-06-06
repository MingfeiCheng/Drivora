from apollo_modules.modules.common.proto.header_pb2 import Header
from apollo_modules.modules.common.proto.geometry_pb2 import PointENU
from apollo_modules.modules.routing.proto.routing_pb2 import LaneWaypoint, RoutingRequest

from .base import Publisher
from ..registry import PUBLISHER_REGISTRY


@PUBLISHER_REGISTRY.register("publisher.routing_request")
class RoutingRequestPublisher(Publisher):
    channel = "/apollo/routing_request"
    msg_type = "apollo.routing.RoutingRequest"
    msg_cls = RoutingRequest
    frequency = 1000.0

    def _process_data(self, message):
        wps = message.waypoints
        # Use lane id+s when known (exact, avoids mis-snapping to a nearby lane
        # that may be excluded from the routing topo graph); fall back to
        # pose+heading only when no lane id is available.
        routing_wps = []
        for wp in wps:
            if wp.lane and wp.lane.id:
                routing_wps.append(LaneWaypoint(id=wp.lane.id, s=wp.lane.s))
            else:
                routing_wps.append(
                    LaneWaypoint(pose=PointENU(x=wp.location.x, y=wp.location.y),
                                 heading=wp.location.yaw)
                )
        return RoutingRequest(
            header=Header(timestamp_sec=message.timestamp, module_name="drivora",
                          sequence_num=self.frame_count),
            waypoint=routing_wps,
        )
