from typing import Any, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image, ImageDraw
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

from typing import Dict, Callable
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import Logger, TensorBoardLogger, WandbLogger
from functools import wraps


from simlingo_base_training.utils.custom_types import DrivingExample
from simlingo_base_training.utils.projection import get_camera_intrinsics, project_points


_STEPS_TO_FIRST_IDX: Dict[int, int] = {}


def once_per_step(function: Callable[[Callback, Trainer, LightningModule, Any, Any, int], None]) -> Callable:
    """
    ``on_train_batch_end`` gets called by pytorch lightning an ``accumulate_grad_batches`` number of times per global step.
    Sometimes ``on_train_batch_end`` is intended per optimisation step, not per each forward pass of a batch.
    This wrapper provides such behaviour, in lack of having found an integrated pytorch lightning way so far*.

    Note:
        Wrapper specifically for `on_train_batch_end` from a pl `Callback`, in regards to the function signature.
    """

    # * `on_before_optimizer_step` is available but gets called before a step is finished, potentially leading to
    #   unexpected behaviour (e.g. report step timings that cut across steps, etc).
    # NOTE: The `_STEPS_TO_FIRST_IDX` global dict is not threadsafe, but `on_train_batch_end` is expected to only be
    #       called sequentially per process.
    # NOTE(technical): When `on_train_batch_end` is called, `trainer.global_step` is already updated. Hence taking the
    # first occurring `batch_idx` for a specific step is effectively the last `batch_idx` from the previous step and the
    # reporting thus is correct.
    @wraps(function)
    def only_on_first_go(
        self: Callback, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        if not all(
            (
                isinstance(self, Callback),
                isinstance(trainer, Trainer),
                isinstance(pl_module, LightningModule),
                isinstance(batch_idx, int),
            )
        ):
            raise ValueError(
                "Only use this decorator on `pl.Callback`'s `on_train_batch_end` function!",
            )
        global_step = trainer.global_step
        if global_step not in _STEPS_TO_FIRST_IDX:
            _STEPS_TO_FIRST_IDX[global_step] = batch_idx

        if _STEPS_TO_FIRST_IDX[global_step] == batch_idx:
            return function(self, trainer, pl_module, outputs, batch, batch_idx)
        return None

    return only_on_first_go


class VisualiseCallback(Callback):
    def __init__(self, interval: int):
        super().__init__()
        self.interval = interval

    @once_per_step
    @torch.no_grad
    def on_train_batch_end(  # pylint: disable=too-many-statements
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: DrivingExample,
        batch_idx: int,
    ):
        if trainer.global_step % self.interval != 0:
            return

        with torch.cuda.amp.autocast(enabled=True):
            # Forward with sampling
            speed_wps, route = pl_module.forward(batch.driving_input)

        try:
            self._visualise_training_examples(batch, speed_wps, trainer, pl_module, 'waypoints')
            self._visualise_training_examples(batch, route, trainer, pl_module, 'route')

            # visualise_cameras(batch, pl_module, trainer, route, speed_wps)
            
            print("visualised_training_example")
            # _LOGGER.info("visualised_training_example")
        except Exception as e:  # pylint: disable=broad-except
            print("visualise_training_examples", e)
            pass
            # _LOGGER.exception("visualise_training_examples", e)
        if hasattr(pl_module, "clear_cache"):
            print("clearing_cache")
            # Clear cache associated with decoding language model
            # _LOGGER.info("clearing_cache")
            pl_module.clear_cache()

    @rank_zero_only
    def _visualise_training_examples(
        self,
        batch: DrivingExample,
        waypoints,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        name: str,
    ):
        if not pl_module.logger:
            return

        # only visualise max 16 examples
        # if len(batch.run_id) > 16:
        if name == 'waypoints':
            waypoint_vis = visualise_waypoints(batch, waypoints)
        elif name == 'route':
            waypoint_vis = visualise_waypoints(batch, waypoints, route=True)
        # pl_module.logger.log_image("visualise/images", images=[Image.fromarray(si_vis)], step=trainer.global_step)
        pl_module.logger.log_image(
            f"visualise/{name}", images=[Image.fromarray(waypoint_vis)], step=trainer.global_step
        )

        plt.close("all")


def fig_to_np(fig):
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

@torch.no_grad()
def visualise_waypoints(batch: DrivingExample, waypoints, route=False):
    assert batch.driving_label is not None

    n = 11
    if route:
        n = 20
    fig = plt.figure(figsize=(10, 10))
    if route:
        gt_waypoints = batch.driving_label.route_adjusted[:, :n, :].cpu().numpy()
    else:
        gt_waypoints = batch.driving_label.poses[..., :n, :2, 3].cpu().numpy()
    pred_waypoints = waypoints[:, :n].cpu().numpy()
    b = gt_waypoints.shape[0]
    # visualise max 16 examples
    b = min(b, 16)
    rows = int(np.ceil(b / 4))
    cols = min(b, 4)


    # add space for text
    fig.subplots_adjust(hspace=0.8)
    for i in range(b):
        ax = fig.add_subplot(rows, cols, i + 1)
        # Predicted waypoints
        ax.scatter(-pred_waypoints[i, :, 1], pred_waypoints[i, :, 0], marker="o", c="b")
        ax.plot(-pred_waypoints[i, :, 1], pred_waypoints[i, :, 0], c="b")
        # Ground truth waypoints (i.e. ideal waypoints)
        ax.scatter(-gt_waypoints[i, :, 1], gt_waypoints[i, :, 0], marker="x", c="g")
        ax.plot(-gt_waypoints[i, :, 1], gt_waypoints[i, :, 0], c="g")
        ax.set_title(f"waypoints {i}")
        ax.grid()
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1.5)

    return fig_to_np(fig)
