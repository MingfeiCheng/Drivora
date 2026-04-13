import carla
import py_trees
import random

from loguru import logger

from tools.timer import GameTime
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ..atomic import AtomicBehavior


class TrafficLightBehaviorConfig(BaseModel):
    pattern: Optional[
        Literal["none", "rule"]
    ] = Field(
        None,
        description=(
            "Traffic light subtype (intersection configuration).\n"
            "- none: Always green (no control).\n"
            "- rule: A rule-based cycle controller."
        ),
    )
    yellow_time: float = Field(..., gt=0, description="Duration of yellow light (seconds)")
    red_time: float = Field(..., gt=0, description="Duration of red light (seconds)")
    green_time: Optional[float] = Field(
        None, gt=0, description="Duration of green light (seconds). If None, defaults to red_time."
    )

    def cycle_length(self) -> float:
        """Return total cycle length."""
        g = self.green_time if self.green_time is not None else self.red_time
        return g + self.yellow_time + self.red_time


class _GroupState:
    """Tracks the cycling state of one junction group."""
    __slots__ = ("active_index", "phase", "phase_start")

    def __init__(self, active_index: int, phase: str, now: float):
        self.active_index = active_index
        self.phase = phase  # "GREEN", "YELLOW", or "RED"
        self.phase_start = now


class TrafficLightBehavior(AtomicBehavior):
    """
    Rule-based traffic light controller.

    Each junction group cycles its subgroups through:
        GREEN(active) -> YELLOW(active) -> RED(active) + GREEN(next) -> ...

    All subgroups except the active one stay RED.
    """

    RED = carla.TrafficLightState.Red
    YELLOW = carla.TrafficLightState.Yellow
    GREEN = carla.TrafficLightState.Green

    def __init__(
        self,
        ctn_operator,
        pattern: str = "rule",
        yellow_time: float = 2.0,
        red_time: float = 1.5,
        green_time: Optional[float] = None,
        debug: bool = False,
        name: str = "TrafficLightBehavior",
    ):
        super().__init__(name)
        self.ctn_operator = ctn_operator
        self.world = self.ctn_operator.get_world()
        self.map = self.world.get_map()
        self.debug = debug

        self.green_time = green_time if green_time is not None else red_time
        self.yellow_time = yellow_time
        self.red_time = red_time

        # Build junction groups once at construction
        self.traffic_lights = list(self.world.get_actors().filter("*traffic_light*"))
        self.groups: List[List[List[carla.TrafficLight]]] = self._build_groups()
        self.group_states: Dict[int, _GroupState] = {}

        # Pre-freeze all lights' internal timers so CARLA doesn't fight us
        self._freeze_timeout = 999999.0

    # ------------------------------------------------------------------
    # Group building (one-time cost)
    # ------------------------------------------------------------------
    def _build_groups(self) -> List[List[List[carla.TrafficLight]]]:
        """Partition traffic lights into junction groups of directional subgroups."""
        groups = []
        visited: set = set()

        for tl in self.traffic_lights:
            if tl.id in visited:
                continue

            annotations = self._annotate_directions(tl)
            subgroups = [annotations[k] for k in ("ref", "opposite", "left", "right") if annotations[k]]

            ids_in_group = {t.id for sg in subgroups for t in sg}
            if ids_in_group:
                groups.append(subgroups)
                visited.update(ids_in_group)

        return groups

    def _annotate_directions(self, traffic_light: carla.TrafficLight) -> Dict[str, List[carla.TrafficLight]]:
        """Classify group members by relative yaw angle."""
        result: Dict[str, List[carla.TrafficLight]] = {"ref": [], "opposite": [], "left": [], "right": []}

        ref_location = self._trigger_location(traffic_light)
        ref_yaw = self.map.get_waypoint(ref_location).transform.rotation.yaw

        for target_tl in traffic_light.get_group_traffic_lights():
            if target_tl.id == traffic_light.id:
                result["ref"].append(target_tl)
                continue

            target_yaw = self.map.get_waypoint(self._trigger_location(target_tl)).transform.rotation.yaw
            diff = (target_yaw - ref_yaw) % 360

            if diff > 330:
                continue  # nearly same direction as ref, skip
            elif diff > 225:
                result["right"].append(target_tl)
            elif diff > 135:
                result["opposite"].append(target_tl)
            elif diff > 30:
                result["left"].append(target_tl)

        return result

    def _trigger_location(self, tl: carla.TrafficLight) -> carla.Location:
        loc = tl.get_transform().transform(tl.trigger_volume.location)
        return carla.Location(loc.x, loc.y, loc.z)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialise(self):
        logger.info(f"[TrafficLightBehavior] {len(self.groups)} junction groups, "
                    f"cycle: G={self.green_time}s Y={self.yellow_time}s R={self.red_time}s")

        now = GameTime.get_time()
        for gid, group in enumerate(self.groups):
            # Pick a random initial active subgroup
            init_idx = random.randint(0, len(group) - 1)
            self._set_subgroup_state(group, init_idx)
            self.group_states[gid] = _GroupState(init_idx, "GREEN", now)

    def _set_subgroup_state(self, group, active_idx, active_state=None):
        """Set one subgroup to active_state (default GREEN), all others to RED. Freeze timers."""
        if active_state is None:
            active_state = self.GREEN
        for idx, subgroup in enumerate(group):
            state = active_state if idx == active_idx else self.RED
            for tl in subgroup:
                tl.set_state(state)
                tl.set_green_time(self._freeze_timeout)
                tl.set_yellow_time(self._freeze_timeout)
                tl.set_red_time(self._freeze_timeout)

    # ------------------------------------------------------------------
    # Per-tick update
    # ------------------------------------------------------------------
    def update(self):
        now = GameTime.get_time()

        for gid, group in enumerate(self.groups):
            gs = self.group_states.get(gid)
            if gs is None:
                continue

            elapsed = now - gs.phase_start

            if gs.phase == "GREEN" and elapsed >= self.green_time:
                # GREEN -> YELLOW for active subgroup
                for tl in group[gs.active_index]:
                    tl.set_state(self.YELLOW)
                gs.phase = "YELLOW"
                gs.phase_start = now

            elif gs.phase == "YELLOW" and elapsed >= self.yellow_time:
                # YELLOW -> RED for active, advance to next subgroup
                prev_idx = gs.active_index
                next_idx = (prev_idx + 1) % len(group)

                # Previous subgroup -> RED
                for tl in group[prev_idx]:
                    tl.set_state(self.RED)

                # Next subgroup -> GREEN
                for tl in group[next_idx]:
                    tl.set_state(self.GREEN)

                gs.active_index = next_idx
                gs.phase = "GREEN"
                gs.phase_start = now

                if self.debug:
                    logger.debug(f"[Group {gid}] Phase {prev_idx} -> {next_idx}")

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        super().terminate(new_status)
        # Reset all lights to green on termination
        for group in self.groups:
            for subgroup in group:
                for tl in subgroup:
                    tl.set_state(self.GREEN)
                    tl.set_green_time(self._freeze_timeout)
