"""
This select route for the ego
Here defines the scenario configuration
"""
import carla

from pydantic import BaseModel, Field
from typing import List, Optional

from .waypoint_vehicle import WaypointVehicleConfig
from .ai_walker import AIWalkerConfig
from .traffic_light import TrafficLightBehaviorConfig
from .static_obstacle import StaticObstacleConfig

from scenario_elements.config import Waypoint

class WeatherConfig(BaseModel):
    pattern: str = Field(..., description="Weather pattern name")
    
    cloudiness: float = Field(..., ge=0, le=100, description="Cloudiness percentage [0, 100]")
    precipitation: float = Field(..., ge=0, le=100, description="Precipitation percentage [0, 100]")
    precipitation_deposits: float = Field(..., ge=0, le=100, description="Precipitation deposits percentage [0, 100]")
    wind_intensity: float = Field(..., ge=0, le=100, description="Wind intensity percentage [0, 100]")
    sun_azimuth_angle: float = Field(..., ge=0, le=360, description="Sun azimuth angle in degrees [0, 360]")
    sun_altitude_angle: float = Field(..., ge=-90, le=90, description="Sun altitude angle in degrees [-90, +90]")
    fog_density: float = Field(..., ge=0, le=100, description="Fog density percentage [0, 100]")
    fog_distance: float = Field(..., ge=0, description="Fog distance in meters [0, inf]")
    wetness: float = Field(..., ge=0, le=100, description="Wetness percentage [0, 100]")
    fog_falloff: float = Field(..., ge=0, description="Fog falloff rate [0, inf]")


    def get_carla_parameters(self) -> carla.WeatherParameters:
        return carla.WeatherParameters(
            cloudiness=self.cloudiness,
            precipitation=self.precipitation,
            precipitation_deposits=self.precipitation_deposits,
            wind_intensity=self.wind_intensity,
            sun_azimuth_angle=self.sun_azimuth_angle,
            sun_altitude_angle=self.sun_altitude_angle,
            fog_density=self.fog_density,
            fog_distance=self.fog_distance,
            wetness=self.wetness,
            fog_falloff=self.fog_falloff
        )

class MapConfig(BaseModel):
    town: str = Field(..., description="Town name, e.g., 'Town03'")
    
    region: Optional[List[float]] = Field(
        None, description="Region in the map defined by [x_min, x_max, y_min, y_max], optional"
    )
    
    
class EgoConfig(BaseModel):
    """Configuration for the ego vehicle."""

    # identity
    id: str = Field(..., description="Unique identifier for the ego vehicle")
    model: Optional[str] = Field(None, description="Ego vehicle model name")
    rolename: str = Field("ego", description="Role name, usually 'ego'")
    color: Optional[str] = Field(
        None, description="Vehicle color in format '(r, g, b)', optional"
    )
    category: Optional[str] = Field(
        "car", description="Actor category, e.g., 'car', 'truck', 'bus'"
    )

    # behavior
    route: List[Waypoint] = Field(
        ..., description="Route as a list of waypoints (x, y, z, pitch, yaw, roll)"
    )
    trigger_time: float = Field(
        0.0, description="Simulation time (seconds) when the ego vehicle starts"
    )
    route_func_coverage: List[str] = Field(
        ..., description="List of task coverage areas, e.g., [('lane type', 'turn left'), ('lane type', 'turn left')]"
    )
    route_esitmated_length: Optional[float] = Field(
        None, description="Estimated route length in meters, optional"
    )

    # agent
    entry_point: Optional[str] = Field(
        None, description="Python entry point for the ADS agent, e.g., 'my_agent:AgentClass'"
    )
    config_path: Optional[str] = Field(
        None, description="Path to the agent configuration file (YAML/JSON)"
    )

class ScenarioConfig(BaseModel):
    id: str = Field(
        ..., 
        description="Unique identifier for the scenario")
    scenario_type: str = Field(
        "open_scenario", 
        description="Type of scenario, e.g., 'intersection', 'lane_change'")
    ego_vehicles: Optional[List[EgoConfig]] = Field(
        None, description="List of Ego vehicle configurations"
    )
    npc_vehicles: Optional[List[WaypointVehicleConfig]] = Field(
        None, description="List of NPC vehicle configurations"
    )
    npc_walkers: Optional[List[AIWalkerConfig]] = Field(
        None, description="List of NPC walker configurations"
    )
    npc_statics: Optional[List[StaticObstacleConfig]] = Field(
        None, description="List of Static obstacle configurations"
    )
    weather: Optional[WeatherConfig] = Field(
        None, description="Weather configuration")
    traffic_light: Optional[TrafficLightBehaviorConfig] = Field(
        None, description="Traffic light behavior configuration")
    map_region: Optional[MapConfig] = Field(
        None, description="Map configuration")
    
    def get_town(self) -> str:
        return self.map_region.town