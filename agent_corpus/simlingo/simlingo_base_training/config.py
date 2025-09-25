from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import time

from hydra.core.config_store import ConfigStore

@dataclass
class LLaVAnextEncoderConfig:
    variant: str = "llava-hf/llava-v1.6-mistral-7b-hf"
    embed_dim: int = 512
    freeze: bool = False
    downsample_feature_grid_factor: Optional[int] = 2
    use_global_img: bool = False

    _target_: str = "simlingo_base_training.models.encoder.llavanext.LLaVAnextEncoderModel"

@dataclass
class ResnetEncoderConfig:
    variant: str = 'microsoft/resnet-34'
    embed_dim: int = 512
    freeze: bool = False
    downsample_feature_grid_factor: Optional[int] = 2
    use_global_img: bool = True

    _target_: str = "simlingo_base_training.models.encoder.resnet.ResnetEncoderModel"


@dataclass
class LanguageModelConfig:
    variant: str = "x-small"
    lora: bool = False
    _target_: str = "simlingo_base_training.models.language_model.llama.Llama"


@dataclass
class DrivingModelConfig:
    vision_model: Any
    language_model: Any

    lr: float = 1e-4
    vision_lr: Optional[float] = 1e-4

    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.999)
    pct_start: float = 0.05
    speed_wps_mode: str = '2d'
    predict_route_as_wps: bool = False
    speed_as_input: bool = True
    new_layer_norm_minmax: bool = False

    _target_: str = "simlingo_base_training.models.driving.DrivingModel"


@dataclass
class DrivingDataModuleConfig:
    batch_size: int = 16
    num_workers: int = 10
    data_path: str = "database/simlingo"
    bucket_path: str = "database/bucketsv2_simlingo"
    encoder: str = "llavanext" # "resnet"
    train_partitions: Optional[Dict[str, float]] = None
    cut_bottom_quarter: bool = False

    use_global_img: bool = False

    skip_first_n_frames: int = 10
    pred_len: int = 11 # including the current time step
    hist_len: int = 3 # including the current time step

    image_enhancing: bool = False
    img_augmentation: bool = True
    img_augmentation_prob: float = 0.5
    img_shift_augmentation: bool = True
    img_shift_augmentation_prob: float = 0.5 # 80% of the data that conains augmented views -> only a small portion of the dataset contains augmented views
    
    num_route_points: int = 20

    use_town13: bool = True
    use_old_towns: bool = True

    route_as: str = 'target_point' # coords, image, target_point
    _target_: str = "simlingo_base_training.dataloader.datamodule.DataModule"


@dataclass
class TrainConfig:
    model: DrivingModelConfig
    data_module: Any

    seed: int = 42
    gpus: int = 8

    resume: bool = False
    resume_path: Optional[str] = None


    debug: bool = False
    overfit: int = 0
    submit: bool = True  # whether to checkpoint and submit the model during training
    fp16_loss_scale: float = 32.0 # 0.0 means dynamic loss scaling, only used with deepspeed

    enable_wandb: bool = True
    wandb_project: Optional[str] = "simlingo_base"
    if debug:
        wandb_name: Optional[str] = f"debug"
        gpus: int = 1
    else:
        # wandb_name: Optional[str] = f"debug"
        name: Optional[str] = "test"
        wandb_name: Optional[str] = f"{time.strftime('%Y_%m_%d_%H_%M_%S')}"
    

    # max_steps: int = 100_000
    max_epochs: int = 20
    precision: str = "16-mixed"
    strategy: str = "deepspeed_stage_2" # deepspeed_stage_2 ddp
    accumulate_grad_batches: int = 1
    devices: Union[str, int] = "auto"
    # val_check_interval: int = 5000
    val_every_n_epochs: int = 1

    checkpoint: Optional[str] = None
    weights: Optional[str] = None  # same as checkpoint, except we don't load optimizer


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="train_base", node=TrainConfig)
    cs.store(group="data_module", name="driving", node=DrivingDataModuleConfig)
    cs.store(group="model", name="driving", node=DrivingModelConfig)
    cs.store(group="model/vision_model", name="llavanext", node=LLaVAnextEncoderConfig)
    cs.store(group="model/vision_model", name="resnet", node=ResnetEncoderConfig)
    cs.store(group="model/language_model", name="llm", node=LanguageModelConfig)


register_configs()
