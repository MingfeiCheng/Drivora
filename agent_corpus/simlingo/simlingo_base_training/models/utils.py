import re
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from simlingo_base_training.utils.custom_types import ParamGroup, TrainingOutput


def configure_params_groups(model: nn.Module, param_groups: Sequence[ParamGroup], verbose: bool = True):
    # Partition the parameters into those that will and those that will not experience regularising weight decay
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.MultiheadAttention, torch.nn.GRU)
    blacklist_weight_modules = (
        LlamaRMSNorm,
        GemmaRMSNorm,
        torch.nn.GroupNorm,
        torch.nn.LayerNorm,
        torch.nn.BatchNorm2d,
        torch.nn.Embedding,
        torch.nn.SyncBatchNorm,
    )

    black_list_weight_names = (
        "latent",
        "logit_scale",
        "pos_enc",
        'image_newline',
        'class_embedding',
        'temporal_embedding',
        "pos_embed",
        "positional_encoding",
        "temporal_encoding",
        "camera_encoding",
        "sink_token",
        "egoposition_head",
        "route_head",
        "acceleration_head",
        "curvature_rate_head",
        "query_embeds",
    )

    decay_params = set()
    no_decay_params = set()
    not_trainable_params = set()
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():

            fpn = f"{mn}.{pn}" if mn else pn  # full param name
            if not p.requires_grad:
                not_trainable_params.add(fpn)
                continue
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay_params.add(fpn)
            elif (
                pn.endswith('weight')
                and isinstance(m, blacklist_weight_modules)
                or any(x in fpn for x in black_list_weight_names)
            ):
                # weights of blacklist modules will NOT be weight decayed
                no_decay_params.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay_params.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay_params & no_decay_params
    union_params = decay_params | no_decay_params | not_trainable_params
    assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
    assert len(param_dict.keys() - union_params) == 0, (
        f"parameters {param_dict.keys() - union_params} were not separated "
        f"into either decay/no_decay/not_trainable set!"
    )

    union_params = set(not_trainable_params)

    # split decay/no decay params into separate param groups with different lr
    out = []
    for pg in param_groups:
        group_params_decay = sorted(k for k in decay_params if re.match(pg.pattern, k))
        group_params_no_decay = sorted(k for k in no_decay_params if re.match(pg.pattern, k))

        inter_params = union_params.intersection(group_params_decay)
        assert len(inter_params) == 0, f"decay parameters {inter_params} made it into multiple param groups"
        inter_params = union_params.intersection(group_params_no_decay)
        assert len(inter_params) == 0, f"no-decay parameters {inter_params} made it into multiple param groups"

        if pg.weight_decay <= 0.0:
            group_params_no_decay = group_params_decay + group_params_no_decay
            group_params_decay = []
        if verbose and rank_zero_only.rank == 0:
            print(f"==== PARAM_GROUP {pg.pattern} ====")
            for pn in group_params_decay:
                print(f"(DECAY) {pn}")
            for pn in group_params_no_decay:
                print(f"(NO_DECAY) {pn}")
        if len(group_params_decay) > 0:
            out.append(
                {'params': [param_dict[pn] for pn in group_params_decay], 'lr': pg.lr, 'weight_decay': pg.weight_decay}
            )
        if len(group_params_no_decay) > 0:
            out.append({'params': [param_dict[pn] for pn in group_params_no_decay], 'lr': pg.lr, 'weight_decay': 0.0})

        union_params.update(group_params_decay)
        union_params.update(group_params_no_decay)

    assert (
        len(param_dict.keys() - union_params) == 0
    ), f"parameters {param_dict.keys() - union_params} were not separated into any param groups!"

    return out


def normalize_imagenet(x):
    """ 
    Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
    x = x.copy()
    x[:, 0] = ((x[:, 0] / 255.0) - 0.485) / 0.229
    x[:, 1] = ((x[:, 1] / 255.0) - 0.456) / 0.224
    x[:, 2] = ((x[:, 2] / 255.0) - 0.406) / 0.225
    return x


def summarise_losses(
    loss_dict: Dict[str, Tuple[Tensor, Tensor]], weights: Optional[Dict[str, float]] = None
) -> TrainingOutput:
    """
    Computes the total loss from a dictionary of losses and their counts.

    The loss dict should contain two tensor for each key:
    - The loss value for each batch sample; shape [B].
    - The loss count for each batch sample; shape [B]. This is the number of items to average over, i.e.
      number of tokens, number of cuboids etc. For the case where each batch sample has a loss, you
      can set it to a ones tensor of shape [B].

    Optionally, a weights dictionary can be provided to weight the losses.

    Args:
        loss_dict: A dictionary of losses and their counts, for each batch sample.
        weights: A dictionary of weights for each loss key.

    Returns:
        A TrainingOutput object with the total loss and its components.
    """

    loss_values = {k: v for k, (v, _) in loss_dict.items()}
    loss_counts = {k: n for k, (_, n) in loss_dict.items()}
    loss_averages = {k: torch.where(n.sum() > 0, v.sum() / n.sum(), 0.0) for k, (v, n) in loss_dict.items()}
    if weights is None:
        loss = torch.stack(list(loss_averages.values())).sum()
    else:
        loss = torch.stack([weights.get(k, 1.0) * v for k, v in loss_averages.items()]).sum()
    return TrainingOutput(
        loss=loss,
        loss_values=loss_values,
        loss_counts=loss_counts,
        loss_averages=loss_averages,
    )