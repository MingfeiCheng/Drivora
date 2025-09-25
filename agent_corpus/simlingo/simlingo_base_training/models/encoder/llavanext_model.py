import torch
from torch.nn import functional as F
from transformers import LlavaNextForConditionalGeneration
from transformers.models.llava_next.modeling_llava_next import (
    LlavaNextCausalLMOutputWithPast,
    get_anyres_image_grid_shape,
    unpad_image,
)
from typing import Optional, Tuple, Union


class LingoLlavaNextModel(LlavaNextForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_global_img = None

    def forward_image(
        self,
        pixel_values: torch.FloatTensor = None,
        image_sizes: Optional[torch.LongTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        downsample_feature_grid_factor: Optional[int] = None,
    ) -> Union[Tuple, LlavaNextCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaNextForConditionalGeneration

        >>> model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

        >>> prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "[INST]  \nWhat is shown in this image? [/INST] The image appears to be a radar chart, which is a type of multi-dimensional plot (...)"
        ```"""

        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if pixel_values is not None:
            batch_size, num_frames, num_cams, num_patches, num_channels, height, width = pixel_values.shape
            reshaped_pixel_values = pixel_values.reshape(batch_size *num_frames * num_cams * num_patches, num_channels, height, width)
            image_features = self.vision_tower(pixel_values=reshaped_pixel_values, output_hidden_states=True)
            
            selected_image_feature = image_features.hidden_states[vision_feature_layer]

            if len(image_features.hidden_states[-1].shape) == 4:
                selected_image_feature = image_features.hidden_states[-1]
                # flatten
                selected_image_feature = selected_image_feature.flatten(2,3)
                # swap
                selected_image_feature = selected_image_feature.transpose(1, 2)

            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:] # remove the CLS token
                # (BS*num_patches), num_tokens->24x24=576, hidden_size=1024... -> 24 because we have 336 pixels and use patch size of 14
            elif vision_feature_select_strategy == "full":
                selected_image_feature = selected_image_feature
                # (BS*num_patches), num_tokens->24x24+1=577, hidden_size=1024

            
            image_features = self.multi_modal_projector(selected_image_feature)

            image_features = list(image_features.view(batch_size * num_frames * num_cams, num_patches, image_features.shape[-2], image_features.shape[-1]))

            # split up image_features for each of the individual images
            # hence we get a list of image_features, each of shape (5, num_patches, hidden_size)
            # if we assume each image has 5 image features (base image + 4 patches)
            # split_sizes = [image.shape[0] for image in pixel_values]


            # image_features = torch.split(image_features, split_sizes, dim=0)

            # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"
            height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size

            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                if image_feature.shape[0] > 1:
                    # image_feature: 
                    if self.use_global_img:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]

                    if height * width != image_feature.shape[1]:
                        raise ValueError("The number of patches is not consistent with the image size.")
                    num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                        image_sizes[image_idx],
                        self.config.image_grid_pinpoints,
                        self.config.vision_config.image_size,
                    )
                    image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1) # torch.Size([1, 3, 24, 24, 4096]) or torch.Size([2, 2, 24, 24, 4096])
                    image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous() # torch.Size([4096, 1, 24, 3, 24]) or torch.Size([4096, 2, 24, 2, 24])
                    image_feature = image_feature.flatten(1, 2).flatten(2, 3)  # torch.Size([4096, 24, 72]) or torch.Size([4096, 48, 48])
                    image_feature = unpad_image(image_feature, image_sizes[image_idx])
                    if downsample_feature_grid_factor is not None:
                        # average pooling to downsample the feature grid
                        image_feature = F.avg_pool2d(
                            image_feature.unsqueeze(0), downsample_feature_grid_factor
                        ).squeeze(0) # for 2: torch.Size([4096, 12, 32])
                        if self.use_global_img:
                            base_image_feature = base_image_feature.view(1, height, width, -1)
                            base_image_feature = F.avg_pool2d(base_image_feature.permute(0, 3, 1, 2), downsample_feature_grid_factor).squeeze(0).permute(1, 2, 0)
                            base_image_feature = base_image_feature.flatten(0, 1)
                    image_feature = torch.cat(
                        (
                            image_feature,
                            self.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1),
                        ),
                        dim=-1,
                    )
                    image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    if self.use_global_img:
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                else:
                    image_feature = image_feature[0]
                    if downsample_feature_grid_factor is not None:
                        image_feature = image_feature.view(1, height, width, -1)
                        image_feature = F.avg_pool2d(image_feature.permute(0, 3, 1, 2), downsample_feature_grid_factor).squeeze(0).permute(1, 2, 0)
                        image_feature = image_feature.flatten(0, 1)
                    image_feature = torch.cat((image_feature, self.image_newline[None]), dim=0)
                new_image_features.append(image_feature)
            image_features = torch.stack(new_image_features, dim=0)

        return image_features
