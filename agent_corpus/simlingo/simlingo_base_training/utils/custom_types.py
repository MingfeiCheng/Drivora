from typing import Dict, List, NamedTuple, Optional
import torch
from torch import Tensor

class DrivingOutput(NamedTuple):
    time_delta_sec: Tensor  # [B, F] float32
    waypoints: Tensor  # [B, F, 2] float32
    # Auxiliary outputs (MUST be at the end):
    language_tokens: Tensor  # [B, max(len(tokens))]
    trajectory_tokens: Tensor  # [B, F, max(len(tokens))]


class TrainingOutput(NamedTuple):
    loss: Tensor  # [] floating

    loss_averages: Dict[str, Tensor]  # [] floating
    loss_values: Dict[str, Tensor]  # [B] floating
    loss_counts: Dict[str, Tensor]  # [B] int64

    driving_output: Optional[DrivingOutput] = None

class ParamGroup(NamedTuple):
    pattern: str
    lr: float
    weight_decay: float

class DrivingInput(NamedTuple):
    camera_images: torch.Tensor  # [B, T, N, C, H, W] uint8 [0, 255]
    image_sizes: torch.Tensor
    camera_intrinsics: torch.Tensor  # [B, N, 3, 3] float32
    camera_extrinsics: torch.Tensor  # [B, N, 4, 4] float32
    vehicle_speed: torch.Tensor  # [B, S] float32 ms
    map_route: torch.Tensor  # [B, 3, RH, RW] uint8 [0, 255]
    target_point: torch.Tensor  # [B, 2] float32

class DrivingLabel(NamedTuple):
    time_delta_sec: Tensor  # [B, F] 0-2 sec
    waypoints: Tensor  # [B, F, 2] 11 future waypoints 0.2s apart
    waypoints_1d: Tensor  # [B, F, 2] 11 future waypoints 0.2s apart
    route_adjusted: Tensor 

class DrivingExample(NamedTuple):
    driving_input: DrivingInput
    driving_label: DrivingLabel
    run_id: List[str]
    timestamp: Tensor  # unix timestamp of ff cam