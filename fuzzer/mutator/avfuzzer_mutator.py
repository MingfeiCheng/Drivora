"""
AVFuzzer Mutator — perturbation-based mutation for GA-driven scenario fuzzing.

Following the AVFuzzer paper methodology, this mutator applies small
perturbations to existing scenario parameters rather than re-sampling
from scratch. Mutable parameters include:

  - NPC vehicle speeds (per waypoint)
  - NPC vehicle trigger times
  - NPC pedestrian trigger times
  - Weather parameters
  - Traffic light timing
"""

import copy
import random

from loguru import logger
from omegaconf import DictConfig

from scenario_corpus.openscenario.config import ScenarioConfig
from scenario_elements.config.waypoint import Waypoint


class AVFuzzerMutator:
    """Perturbation-based scenario mutator for AVFuzzer."""

    def __init__(self, config: DictConfig):
        self.config = config

        # Perturbation magnitudes (configurable via YAML)
        self.speed_delta = config.get('speed_delta', 2.0)          # m/s
        self.speed_min = config.get('speed_min', 1.0)
        self.speed_max = config.get('speed_max', 15.0)
        self.trigger_delta = config.get('trigger_delta', 2.0)      # seconds
        self.trigger_min = config.get('trigger_min', 0.0)
        self.trigger_max = config.get('trigger_max', 10.0)
        self.weather_delta = config.get('weather_delta', 15.0)     # percentage points
        self.tl_time_delta = config.get('tl_time_delta', 2.0)     # seconds

    def perturb(self, scenario: ScenarioConfig) -> ScenarioConfig:
        """Apply random perturbations to a scenario's mutable parameters.

        Each sub-mutation is applied with probability 0.5, so not every
        parameter changes every generation — this gives the GA a mix of
        exploration and exploitation.
        """
        scenario = copy.deepcopy(scenario)

        if scenario.npc_vehicles and random.random() < 0.8:
            scenario = self._perturb_npc_vehicles(scenario)

        if scenario.npc_walkers and random.random() < 0.5:
            scenario = self._perturb_walkers(scenario)

        if scenario.weather and random.random() < 0.5:
            scenario = self._perturb_weather(scenario)

        if scenario.traffic_light and random.random() < 0.3:
            scenario = self._perturb_traffic_light(scenario)

        return scenario

    # ── NPC vehicle perturbation ──────────────────────────────────────────

    def _perturb_npc_vehicles(self, scenario: ScenarioConfig) -> ScenarioConfig:
        for npc in scenario.npc_vehicles:
            # Perturb trigger time
            if random.random() < 0.5:
                npc.trigger_time = self._clamp(
                    npc.trigger_time + random.uniform(-self.trigger_delta, self.trigger_delta),
                    self.trigger_min, self.trigger_max)

            # Perturb waypoint speeds
            if random.random() < 0.7:
                for wp in npc.route:
                    wp.speed = self._clamp(
                        wp.speed + random.uniform(-self.speed_delta, self.speed_delta),
                        self.speed_min, self.speed_max)

        return scenario

    # ── Pedestrian perturbation ───────────────────────────────────────────

    def _perturb_walkers(self, scenario: ScenarioConfig) -> ScenarioConfig:
        for walker in scenario.npc_walkers:
            # Perturb trigger time
            walker.trigger_time = self._clamp(
                walker.trigger_time + random.uniform(-self.trigger_delta, self.trigger_delta),
                self.trigger_min, self.trigger_max)

        return scenario

    # ── Weather perturbation ──────────────────────────────────────────────

    def _perturb_weather(self, scenario: ScenarioConfig) -> ScenarioConfig:
        w = scenario.weather
        d = self.weather_delta

        w.cloudiness = self._clamp(w.cloudiness + random.uniform(-d, d), 0, 100)
        w.precipitation = self._clamp(w.precipitation + random.uniform(-d, d), 0, 100)
        w.precipitation_deposits = self._clamp(w.precipitation_deposits + random.uniform(-d, d), 0, 100)
        w.wind_intensity = self._clamp(w.wind_intensity + random.uniform(-d, d), 0, 100)
        w.fog_density = self._clamp(w.fog_density + random.uniform(-d, d), 0, 100)
        w.fog_distance = self._clamp(w.fog_distance + random.uniform(-d, d), 0, 200)
        w.wetness = self._clamp(w.wetness + random.uniform(-d, d), 0, 100)
        w.sun_altitude_angle = self._clamp(
            w.sun_altitude_angle + random.uniform(-10, 10), -90, 90)

        return scenario

    # ── Traffic light perturbation ────────────────────────────────────────

    def _perturb_traffic_light(self, scenario: ScenarioConfig) -> ScenarioConfig:
        tl = scenario.traffic_light
        d = self.tl_time_delta

        tl.green_time = max(2.0, tl.green_time + random.uniform(-d, d)) if tl.green_time else None
        tl.yellow_time = max(1.0, tl.yellow_time + random.uniform(-d, d))
        tl.red_time = max(2.0, tl.red_time + random.uniform(-d, d))

        return scenario

    # ── Utility ───────────────────────────────────────────────────────────

    @staticmethod
    def _clamp(value, lo, hi):
        return max(lo, min(hi, value))
