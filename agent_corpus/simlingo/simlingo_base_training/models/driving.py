import pickle as pkl
from pprint import PrettyPrinter
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
from deepspeed.ops.adam import FusedAdam
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18

from simlingo_base_training.models.adaptors.adaptors import (
    AdaptorList, DrivingAdaptor, VectorInputAdaptor, WaypointInputAdaptor
)
from simlingo_base_training.models.utils import configure_params_groups, summarise_losses
from simlingo_base_training.utils.custom_types import (
    DrivingExample, DrivingInput, DrivingLabel, ParamGroup, TrainingOutput
)

pprint = PrettyPrinter().pprint

class RouteEncode(nn.Module):
    def __init__(self, out_channels: int, pretrained=True):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, out_channels)

    def forward(self, route):
        x = route.to(self.backbone.fc.weight.dtype) / 128.0 - 1.0
        return self.backbone(x).unsqueeze(-2)  # add token dim

class NormZeroOne(nn.Module):
    def __init__(self, min_max: Tuple[float, float]):
        super().__init__()
        self.register_buffer("min_max", torch.tensor(min_max, dtype=torch.float), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Normalise tensor to [0, 1] using values from min_max"""
        return (x - self.min_max[0]) / (self.min_max[1] - self.min_max[0])


class DrivingModel(pl.LightningModule):
    def __init__(
        self,
        vision_model: nn.Module,
        language_model: nn.Module,
        lr: float = 1e-4,
        vision_lr: Optional[float] = None,
        weight_decay: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.999),
        pct_start: float = 0.05,
        enable_language=False,
        route_as=True,
        speed_as_input=True,
        new_layer_norm_minmax=False,
        predict_route_as_wps=False,
        speed_wps_mode=False,
        variant=None,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.vision_model = vision_model
        self.language_model = language_model
        self.lr = lr
        self.vision_lr = vision_lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.pct_start = pct_start
        self.enable_language = enable_language
        self.route_as = route_as
        self.speed_as_input = speed_as_input
        self.new_layer_norm_minmax = new_layer_norm_minmax
        self.predict_route_as_wps = predict_route_as_wps
        self.speed_wps_mode = speed_wps_mode

        self.all_predictions = {}
        self.all_losses = {}

        driving = DrivingAdaptor(
            self.language_model.hidden_size, 
            speed_wps_mode=speed_wps_mode,
            predict_route_as_wps=predict_route_as_wps
        )
        
        self.adaptors = AdaptorList(
            driving=driving,
        )

        if self.speed_as_input:
            if self.new_layer_norm_minmax:
                min_max = (0.0, 110.0 / 3.6)
            else:
                min_max = (0.0, 64.0 / 3.6)
            self.speed_encoder = VectorInputAdaptor(
                input_size=1,
                token_size=self.language_model.hidden_size,
                hidden_size=256,
                norm_layer=NormZeroOne(min_max=min_max),
            )

        if route_as == 'coords' or route_as == 'target_point':
            if self.new_layer_norm_minmax:
                min_max = (-200.0, 200.0)
            else:
                min_max = (-32.0, 32.0)
            self.route_encoder = WaypointInputAdaptor(
                token_size=self.language_model.hidden_size,
                hidden_size=256,
                norm_layer=NormZeroOne(min_max=min_max),
            )
        else:
            self.route_encoder = RouteEncode(self.language_model.hidden_size, pretrained=True)

        self.language_projection = nn.Identity()
        if self.vision_model.token_size != self.language_model.hidden_size:
            self.language_projection = nn.Linear(self.vision_model.token_size, self.language_model.hidden_size, bias=False)


        self.tok = self.language_model.tokenizer
        self.bos_token_id = self.tok.bos_token_id
        self.eos_token_id = self.tok.eos_token_id
        self.pad_token_id = self.tok.pad_token_id

    def forward(self,
        driving_input: DrivingInput,
        prompt_ids: Optional[Tensor] = None):
        """
        Samples a trajectory from the model.
        """
        self.speed_wps, self.route, self.target_speed = None, None, None

        BS = driving_input.camera_images.size(0)
        input_embeds, _ = self.get_fixed_input_embeds(driving_input)

        # single forward pass same as during training so we can use the same function
        inputs = self.adaptors(driving_input)
        features = self.forward_model(driving_input, inputs["inputs"])
        predictions = self.adaptors.driving.get_predictions(features)

        for k, v in predictions.items():
            if v is not None:
                setattr(self, k, v)

        return self.speed_wps, self.route


    def forward_model(self, 
                      driving_input: DrivingInput, 
                      adaptor_embeds: Tensor, 
                      driving_labels: DrivingLabel = None,
                    #   language_embeds: Tensor = None
                      ) -> Tensor:
        """
        Forward model conditioned on the given driving input.
        """

        vision_embeds, vision_attention_mask = self.get_fixed_input_embeds(driving_input)

        input_embeds = torch.cat((vision_embeds, adaptor_embeds), dim=1)
        # to dtype of language model
        input_embeds = input_embeds.to(
            dtype=self.language_model.model.dtype
        )

        outputs = self.language_model.forward(
            input_embeds,
        )

        vision_outputs, adaptor_outputs = outputs.split(
            [outputs.size(1) - adaptor_embeds.size(1), adaptor_embeds.size(1)], dim=1
        )
        return adaptor_outputs
    
    def get_fixed_input_embeds(self, driving_input: DrivingInput):
        img = driving_input.camera_images #[:, 0, :, :, :] # only use the front camera
        map_route = driving_input.map_route

        vision_embeds, _ = self.vision_model.forward(img, image_sizes=driving_input.image_sizes)
        attention_mask = None

        # n_frames, n_tokens, channels = sizes
        vision_embeds = self.language_projection(vision_embeds)
        # channels = vision_embeds.size(2)
        BS = vision_embeds.size(0)
        route = self.route_encoder.forward(map_route)
        if self.speed_as_input:
            speed = self.speed_encoder.forward(driving_input.vehicle_speed)

            input_embeds = torch.cat((vision_embeds, speed, route), dim=1)
        else:
            input_embeds = torch.cat((vision_embeds, route), dim=1)

        return input_embeds, attention_mask

    def forward_loss(self, example: DrivingExample, per_sample=False) -> TrainingOutput:
        """
        Forward pass of the model for a driving input, followed by
        computing the next token cross-entropy loss.

        Args:
            driving_input: input to the vision encoder.
            text_ids: Text ids tensor of shape [B, T]. These are input to the model and used in the loss.
            text_mask: Text mask tensor of shape [B, T].
        """

        adaptor_dict = self.adaptors(example)
        adaptor_embeds = adaptor_dict["inputs"]

        adaptor_outputs = self.forward_model(example.driving_input, adaptor_embeds, driving_labels=example.driving_label)
        loss_dict = self.adaptors.compute_loss(adaptor_outputs, adaptor_dict, example)

        loss_dict_only_losses = {k:v for k, v in loss_dict.items() if k.endswith("loss")}
        pred_labels = {k:v for k, v in loss_dict.items() if not k.endswith("loss")}
        if per_sample:
            return loss_dict_only_losses, pred_labels

        return summarise_losses(loss_dict_only_losses)

    def training_step(self, batch: DrivingExample, _batch_idx: int = 0):
        output = self.forward_loss(batch)
        self.log_training_output(output, "train")

        # log the loss
        self.log("train/loss", output.loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": output.loss, "outputs": output}

    def validation_step(self, batch: DrivingExample, _batch_idx: int = 0):
        output = self.forward_loss(batch)
        self.log_training_output(output, "val")

        # log the loss
        self.log("val/loss", output.loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": output.loss, "outputs": output}

    def predict_step(self, batch: DrivingExample, _batch_idx: int = 0):
        loss_dict, pred_labels = self.forward_loss(batch, per_sample=True)

        per_sample_losses = loss_dict['waypoints_loss'][0].detach().cpu().numpy()
        predictions = pred_labels['waypoints_prediction'].detach().cpu().numpy()
        labels = pred_labels['waypoints_label'].detach().cpu().numpy()

        for i in range(len(per_sample_losses)):
            self.all_losses[batch.run_id[i]] = per_sample_losses[i]
            self.all_predictions[batch.run_id[i]] = (per_sample_losses[i], predictions[i], labels[i])

        return

    def on_predict_epoch_end(self) -> None:

        # sort by loss and save as pkl and json
        # sorted_losses = sorted(self.all_losses.items(), key=lambda x: x[1])
        # with open("sorted_losses.json", "w") as f:
        #     json.dump(sorted_losses, f)
        try:
            with open("sorted_losses.pkl", "wb") as f:
                pkl.dump(self.all_losses, f)
            with open("all_predictions.pkl", "wb") as f:
                pkl.dump(self.all_predictions, f)
        except:
            breakpoint()


    def log_training_output(self, training_output: TrainingOutput, mode: str, dataset: Optional[str] = None):
        losses = {k: n.detach() for k, n in training_output.loss_averages.items()}
        counts = {k: n.detach().sum() for k, n in training_output.loss_counts.items()}
        losses["loss"] = training_output.loss.detach()
        counts["loss"] = 1  # loss is already averaged
        for k, v in sorted(losses.items()):
            log_key = f"{mode}_losses/{k}"
            self.log(log_key, v, batch_size=counts[k], sync_dist=True, add_dataloader_idx=False)


    def configure_optimizers(self):

        param_groups = [
            ParamGroup(r"^(model|language_model|language_projection|adaptors|speed_encoder|route_encoder)\..*", self.lr, self.weight_decay),
            ParamGroup(r"^vision_model\..*", self.vision_lr, self.weight_decay),
        ]
        optimizer_class = (
            FusedAdam if isinstance(self.trainer.strategy, pl.strategies.DeepSpeedStrategy) else torch.optim.AdamW
        )
        optimizer = optimizer_class(configure_params_groups(self, param_groups, verbose=False), betas=self.betas)
        lrs = [pg['lr'] for pg in optimizer.param_groups]
        if self.trainer.max_steps == -1:
            max_steps = self.trainer.estimated_stepping_batches
        else:
            max_steps = self.trainer.max_steps
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lrs, total_steps=max_steps, pct_start=self.pct_start
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "frequency": 1, "interval": "step"}}