from typing import Dict, Any
from dataclasses import dataclass, field, fields, replace

from fuzzer.open_scenario.base import FuzzSeed

@dataclass
class BehSeed(FuzzSeed):
    is_new: bool = field(default=False)          # whether it has new coverage
    parent_index: int = field(default=-1)        # index in all_seeds
    select_num: int = field(default=0)
    fail_num: int = field(default=0)
    safety_score: float = field(default=1.0)     # [0.0, 1.0], lower is better
    diversity_score: float = field(default=0.0)  # [0.0, 1.0], higher is better
    energy: float = field(default=1.0)           # [0.0, 1.0], higher is better
    scenario_dir: str = field(default="")        # directory where the scenario is stored

    @classmethod
    def load_from_scenario_file(cls, config_path: str) -> "BehSeed":
        base = super().load_from_scenario_file(config_path)
        # upgrade FuzzSeed instance into BehSeed by replacing/adding fields
        return replace(base)

    @classmethod
    def load_from_dict(cls, data: Dict[str, Any]) -> "BehSeed":
        base = super().load_from_dict(data)
        # collect extra fields defined in BehSeed but not in FuzzSeed
        extra = {
            f.name: data.get(f.name, getattr(cls, f.name))
            for f in fields(cls)
            if f.name not in {f.name for f in fields(FuzzSeed)}
        }
        return replace(base, **extra)

    def to_dict(self) -> dict:
        data = super().to_dict()
        # automatically include subclass-specific fields
        for f in fields(self):
            if f.name not in data:
                data[f.name] = getattr(self, f.name)
        return data