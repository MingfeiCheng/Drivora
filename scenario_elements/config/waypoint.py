from pydantic import BaseModel, Field
from typing import Optional

class Waypoint(BaseModel):
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    z: float = Field(..., description="Z coordinate")
    pitch: float = Field(..., description="Pitch angle in degrees")
    yaw: float = Field(..., description="Yaw angle in degrees")
    roll: float = Field(..., description="Roll angle in degrees")
    speed: float = Field(..., ge=0, description="Speed at this waypoint")
    road_option: Optional[str] = Field(None, description="Road option at this waypoint, e.g., 'LANE_FOLLOW', 'LEFT', 'RIGHT', 'STRAIGHT', 'CHANGE_LANE_LEFT', 'CHANGE_LANE_RIGHT', 'LANE_KEEP'")