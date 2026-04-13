import os
import json
import numpy as np

from loguru import logger
from dataclasses import dataclass
from typing import Optional, List, Dict
from omegaconf import DictConfig

from scenario_runner.ctn_operator import get_carla_version, OLD_CARLA_VERSION

TASK_SCOPE = [
    'LANEFOLLOW',
    'LEFT',
    'RIGHT',
    'STRAIGHT',
    'CHANGELANELEFT',
    'CHANGELANERIGHT'
]

@dataclass
class MapRegionSpace:
    
    town: str = "Town02"
    region_x_min: Optional[float] = -np.inf
    region_x_max: Optional[float] = np.inf
    region_y_min: Optional[float] = -np.inf
    region_y_max: Optional[float] = np.inf
    
    map_driving_waypoints_cache: Optional[List[List[float]]] = None
    map_ego_routes_cache: Optional[Dict[str, List[List[float]]]] = None
    map_npc_routes_cache: Optional[Dict[str, List[List[float]]]] = None
    
    cache_npc_route_list: Optional[List[List[float]]] = None
    cache_ego_route_list: Optional[List[List[float]]] = None
    
    def build_route_indexes(self, task_name):
        """Build all route indexes."""
        self.build_npc_route_index()
        self.build_ego_route_index(task_name)
    
    def build_npc_route_index(self):
        """Build route_name → route_id mapping from dict keys."""
        assert self.map_npc_routes_cache is not None, "npc routes cache is None!"
        
        route_names = sorted(list(self.map_npc_routes_cache.keys()))
        # convert to list of routes
        npc_route_list = []
        for name in route_names:
            npc_route_list.extend(self.map_npc_routes_cache[name])
        
        if self.cache_npc_route_list is None:
            self.cache_npc_route_list = npc_route_list    
                
    def build_ego_route_index(self, task_name):
        """Build route_name → route_id mapping from dict keys."""
        assert self.map_ego_routes_cache is not None, "ego routes cache is None!"
        
        if task_name is not None:
            # logger.debug(f"Build ego route index for task: {task_name} existing routes: {list(self.map_ego_routes_cache.keys())}")
            if task_name != 'LANEFOLLOW':
                for key in self.map_ego_routes_cache.keys():
                    if task_name in key:
                        task_name = key
                        break
                    
            ego_route_list = self.map_ego_routes_cache[task_name]
            if self.cache_ego_route_list is None:
                self.cache_ego_route_list = ego_route_list
        else:
            route_names = sorted(list(self.map_ego_routes_cache.keys()))
            # convert to list of routes
            ego_route_list = []
            for name in route_names:
                ego_route_list.extend(self.map_ego_routes_cache[name])
            
            if self.cache_ego_route_list is None:
                self.cache_ego_route_list = ego_route_list

    # -------------------------------
    # SAVE: simple dump of dataclass
    # -------------------------------
    def save_to_file(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    # -------------------------------
    # LOAD: robust, future-proof
    # -------------------------------
    def load_from_file(self, filepath: str):
        with open(filepath, "r") as f:
            data = json.load(f)

        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"[WARNING] MapRegionSpace: unknown field '{key}' in {filepath}, ignored.")

    
@dataclass
class EGOSpace:
    
    num = [1, 1]
    trigger_time = [0.0, 5.0]
    route_length = [50.0, 200.0]
    region_buffer = 1.0 # in meters
    corssing_junction = True
    
    # other hyperparameters
    dist2vehicle_same_lane = 10.0
    dist2vehicle_other_lane = 2.0 # in case overlapping
    dist2pedestrian = 1.0
    dist2static_same_lane = 10.0
    dist2static_other_lane = 5.0
    
    if get_carla_version() < OLD_CARLA_VERSION:
        models = [
            "vehicle.lincoln.mkz2017"
        ]
    else:
        models = [
            "vehicle.lincoln.mkz_2017"
        ]
        
    # configurable parameters for seeds
    # note: if this is fixed, the ego vehicle will always spawn at the same location
    ego_start_point: Optional[List[float]] = None  # list of spawn points
    ego_end_point: Optional[List[float]] = None    # list of end points
    ego_task_assignment: Optional[str] = None  # e.g., "follow_lane", "reach_destination", "cross_junction"
    

@dataclass
class NPCVehicleSpace:
    
    # npc vehicles
    trigger_time = [0.0, 5.0]
    num = [5, 5] # now we fixed
    speed = [2.0, 10.0]  # in m/s
    delta_speed = [0.0, 2.0] # in m/s
    route_length = [100.0, 400.0]  # in meters
    # other hyperparameters
    dist2vehicle_same_lane = 10.0
    dist2vehicle_other_lane = 2.0 # in case overlapping
    dist2pedestrian = 1.0 # always on sidewals
    dist2static_same_lane = 10.0
    dist2static_other_lane = 5.0
    
    if get_carla_version() < OLD_CARLA_VERSION:
        models = [
            "vehicle.audi.a2",
            "vehicle.audi.etron",
            "vehicle.audi.tt",
            "vehicle.bh.crossbike",
            "vehicle.bmw.grandtourer",
            "vehicle.bmw.isetta",
            "vehicle.carlamotors.carlacola",
            "vehicle.chevrolet.impala",
            "vehicle.citroen.c3",
            "vehicle.diamondback.century",
            "vehicle.dodge_charger.police",
            "vehicle.gazelle.omafiets",
            "vehicle.harley-davidson.low_rider",
            "vehicle.jeep.wrangler_rubicon",
            "vehicle.kawasaki.ninja",
            "vehicle.lincoln.mkz2017",
            "vehicle.mercedes-benz.coupe",
            "vehicle.mini.cooperst",
            "vehicle.mustang.mustang",
            "vehicle.nissan.micra",
            "vehicle.nissan.patrol",
            "vehicle.seat.leon",
            "vehicle.tesla.cybertruck",
            "vehicle.tesla.model3",
            "vehicle.toyota.prius",
            "vehicle.volkswagen.t2",
            "vehicle.yamaha.yzf",
        ]
    else:
        models = [
            "vehicle.audi.a2",
            "vehicle.audi.etron",
            "vehicle.audi.tt",
            "vehicle.bh.crossbike",
            "vehicle.bmw.grandtourer",
            "vehicle.carlamotors.carlacola",
            "vehicle.carlamotors.firetruck",
            "vehicle.chevrolet.impala",
            "vehicle.citroen.c3",
            "vehicle.diamondback.century",
            "vehicle.dodge.charger_2020",
            "vehicle.dodge.charger_police",
            "vehicle.dodge.charger_police_2020",
            "vehicle.ford.ambulance",
            "vehicle.ford.crown",
            "vehicle.ford.mustang",
            "vehicle.gazelle.omafiets",
            "vehicle.harley-davidson.low_rider",
            "vehicle.jeep.wrangler_rubicon",
            "vehicle.kawasaki.ninja",
            "vehicle.lincoln.mkz_2017",
            "vehicle.lincoln.mkz_2020",
            "vehicle.mercedes.coupe",
            "vehicle.mercedes.coupe_2020",
            "vehicle.mercedes.sprinter",
            "vehicle.micro.microlino",
            "vehicle.mini.cooper_s",
            "vehicle.mini.cooper_s_2021",
            "vehicle.nissan.micra",
            "vehicle.nissan.patrol",
            "vehicle.nissan.patrol_2021",
            "vehicle.seat.leon",
            "vehicle.tesla.cybertruck",
            "vehicle.tesla.model3",
            "vehicle.toyota.prius",
            "vehicle.vespa.zx125",
            "vehicle.volkswagen.t2",
            "vehicle.volkswagen.t2_2021",
            "vehicle.yamaha.yzf",
        ]
    
@dataclass
class NPCPedestrianSpace:
        
    # npc pedestrians - AI controller
    num = [1, 5] # now we fixed
    trigger_time = [0.0, 5.0]

    # other hyperparameters
    dist2vehicle = 1.0
    dist2pedestrian = 0.5 # always on sidewals
    dist2static = 2.0
    
    if get_carla_version() < OLD_CARLA_VERSION:
        models = [
            "walker.pedestrian.0001",
            "walker.pedestrian.0002",
            "walker.pedestrian.0003",
            "walker.pedestrian.0004",
            "walker.pedestrian.0005",
            "walker.pedestrian.0006",
            "walker.pedestrian.0007",
            "walker.pedestrian.0008",
            "walker.pedestrian.0009",
            "walker.pedestrian.0010",
            "walker.pedestrian.0011",
            "walker.pedestrian.0012",
            "walker.pedestrian.0013",
            "walker.pedestrian.0014",
            "walker.pedestrian.0015",
        ]
    else:
        models = [
            "walker.pedestrian.0001",
            "walker.pedestrian.0002",
            "walker.pedestrian.0003",
            "walker.pedestrian.0004",
            "walker.pedestrian.0005",
            "walker.pedestrian.0006",
            "walker.pedestrian.0007",
            "walker.pedestrian.0008",
            "walker.pedestrian.0009",
            "walker.pedestrian.0010",
            "walker.pedestrian.0011",
            "walker.pedestrian.0012",
            "walker.pedestrian.0013",
            "walker.pedestrian.0014",
            "walker.pedestrian.0015",
            "walker.pedestrian.0016",
            "walker.pedestrian.0017",
            "walker.pedestrian.0018",
            "walker.pedestrian.0019",
            "walker.pedestrian.0020",
            "walker.pedestrian.0021",
            "walker.pedestrian.0022",
            "walker.pedestrian.0023",
            "walker.pedestrian.0024",
            "walker.pedestrian.0025",
            "walker.pedestrian.0026",
            "walker.pedestrian.0027",
            "walker.pedestrian.0028",
            "walker.pedestrian.0029",
            "walker.pedestrian.0030",
            "walker.pedestrian.0031",
            "walker.pedestrian.0032",
            "walker.pedestrian.0033",
            "walker.pedestrian.0034",
            "walker.pedestrian.0035",
            "walker.pedestrian.0036",
            "walker.pedestrian.0037",
            "walker.pedestrian.0038",
            "walker.pedestrian.0039",
            "walker.pedestrian.0040",
            "walker.pedestrian.0041",
            "walker.pedestrian.0042",
            "walker.pedestrian.0043",
            "walker.pedestrian.0044",
            "walker.pedestrian.0045",
            "walker.pedestrian.0046",
            "walker.pedestrian.0047",
            "walker.pedestrian.0048",
            "walker.pedestrian.0049",
        ]

@dataclass
class NPCStaticSpace:
    # only on road
        
    num = [0, 2]
    
    # other hyperparameters
    dist2vehicle_same_lane = 10.0
    dist2vehicle_other_lane = 5.0 # in case overlapping
    dist2pedestrian = 1.0 # always on sidewals
    dist2static_same_lane = 5.0
    dist2static_other_lane = 15.0
    
    
    models = [
        "static.prop.streetbarrier",
        "static.prop.constructioncone",
        "static.prop.trafficcone01",
        "static.prop.trafficcone02",
        "static.prop.trafficwarning"
    ]
    
@dataclass
class WeatherSpace:
    """Full valid CARLA weather parameter ranges."""

    cloudiness = [0.0, 100.0]               # percentage
    precipitation = [0.0, 100.0]            # percentage
    precipitation_deposits = [0.0, 100.0]   # percentage
    wind_intensity = [0.0, 100.0]           # percentage
    sun_azimuth_angle = [0.0, 360.0]        # degree
    sun_altitude_angle = [-90.0, 90.0]      # degree (can go below horizon)
    fog_density = [0.0, 100.0]              # percentage
    fog_distance = [0.0, 300.0]             # percentage
    fog_falloff = [0.0, 5.0]              # percentage
    wetness = [0.0, 100.0]                  # percentage
    
    ratio_wetness_precipitation = 0.7  # wetness should be at least equal to precipitation
    ratio_precipitation_deposits_precipitation = 0.5  # deposits should be at least half of precipitation
    cloudiness_sun_threshold = 80  # when cloudiness is above this, sun altitude should be limited
    cloudiness_sun_altitude_angle = 30  # when cloudiness is 100%, sun altitude should be at most 0 degree
    wind_intensity_threshold = 70  # when wind intensity is above this, more clouds are expected
    cloudiness_intensity = 40  # when wind intensity is high, cloudiness should be at least this value
    
@dataclass
class TrafficLightSpace:
    """Traffic light states."""

    pattern = ["none", "rule"] # NOTE: we first only using green lights, for quick safety check
    
    green_duration = [5.0, 10.0]  # in seconds
    yellow_duration = [2.0, 3.0]  # in seconds
    red_duration = [5.0, 10.0]    # in seconds


class ScenarioODDSpace:
    """Overall ODD space for scenario sampling."""
    
    map_region_space: MapRegionSpace = MapRegionSpace()
    ego_space: EGOSpace = EGOSpace()
    npc_vehicle_space: NPCVehicleSpace = NPCVehicleSpace()
    npc_pedestrian_space: NPCPedestrianSpace = NPCPedestrianSpace()
    npc_static_space: NPCStaticSpace = NPCStaticSpace()
    weather_space: WeatherSpace = WeatherSpace()
    traffic_light_space: TrafficLightSpace = TrafficLightSpace()

    def __init__(self, space_config: Optional[DictConfig] = None):
        """
        space_config 是一个 OmegaConf DictConfig，用法：
        
        space_config:
            ego_space:
                trigger_time: [0.0, 10.0]
                route_length: [40, 300]
            weather_space:
                precipitation: [0, 80]
                fog_distance: [0, 200]
        
        只更新给定字段，未指定字段保持默认。
        """

        if space_config is None:
            return
        
        # 依次更新每个空间
        self._assign_config_to_space("map_region_space", MapRegionSpace, space_config)
        self._assign_config_to_space("ego_space", EGOSpace, space_config)
        self._assign_config_to_space("npc_vehicle_space", NPCVehicleSpace, space_config)
        self._assign_config_to_space("npc_pedestrian_space", NPCPedestrianSpace, space_config)
        self._assign_config_to_space("npc_static_space", NPCStaticSpace, space_config)
        self._assign_config_to_space("weather_space", WeatherSpace, space_config)
        self._assign_config_to_space("traffic_light_space", TrafficLightSpace, space_config)

    def _assign_config_to_space(self, attr_name: str, cls, cfg: DictConfig):
        """
        将 cfg 中 attr_name 对应的配置赋值给 self.attr_name。

        attr_name : 字段名称（字符串）
        cls : dataclass 类
        cfg : DictConfig
        
        示例：
        cfg = {
          "ego_space": {
              "trigger_time": [0, 5],
              "route_length": [60, 200]
          }
        }
        """

        if attr_name not in cfg:
            return
        
        sub_cfg = cfg[attr_name]
        space_obj = getattr(self, attr_name)

        for key, val in sub_cfg.items():
            if hasattr(space_obj, key):
                setattr(space_obj, key, val)
            else:
                logger.warning(f"[WARNING] Unknown config key '{key}' in '{attr_name}' (ignored).")
