import numpy as np
import torch
from torch import nn
from transformers import LlavaNextProcessor

from simlingo_base_training.models.encoder.llavanext_model import LingoLlavaNextModel

class LLaVAnextEncoderModel(nn.Module):
    def __init__(self,
        variant: str, 
        embed_dim: int,
        freeze: bool,
        downsample_feature_grid_factor: int = 2,
        use_global_img = False,
    ):
    
        super().__init__()
        self.num_cameras = 1
        self.num_frames = 1
        self.token_size = embed_dim

        self.downsample_feature_grid_factor = downsample_feature_grid_factor

        self.image_encoder = LingoLlavaNextModel.from_pretrained(variant)
        self.image_encoder.use_global_img = use_global_img
        self.image_encoder.config.image_grid_pinpoints = [[336,672]] # this is done to save memory otherwise it would use higehr res input dependen on how we cut the image
        self.image_encoder.language_model = None
        print("\033[91m" + "Using LLaVA pretraining for the image encoder" + "\033[0m")
        
        self.projection = nn.Linear(self.image_encoder.base_model.config.vision_config.intermediate_size, embed_dim)
        # Embeddings: BS, N_FRAMES, N_CAMS, N_PATCHES, EMBED_DIM
        self.temporal_encoding = nn.Parameter(0.02 * torch.randn(1, self.num_frames, 1, 1, embed_dim))
        self.camera_encoding = nn.Parameter(0.02 * torch.randn(1, 1, self.num_cameras, 1, embed_dim))
        
        # freeze the paramaeters -> no gradient updates
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

            # activate the projection layer
            self.projection.weight.requires_grad = True
            self.projection.bias.requires_grad = True
            # self.positional_encoding.requires_grad = True
            self.temporal_encoding.requires_grad = True
            self.camera_encoding.requires_grad = True

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_sizes = None,
        use_temporal_encoding: bool = True,
        use_positional_encoding: bool = True,
        use_camera_encoding: bool = True,
    ) -> torch.Tensor:

        BS, num_frames, num_cams, num_patches, C, H, W = pixel_values.shape


        patch_embeddings = self.image_encoder.forward_image(pixel_values=pixel_values, image_sizes=image_sizes, downsample_feature_grid_factor=self.downsample_feature_grid_factor)
        patch_embeddings = self.projection(patch_embeddings)
        patch_embeddings = patch_embeddings.view(-1, num_frames, num_cams, patch_embeddings.shape[-2], patch_embeddings.shape[-1])

        input_sequence = patch_embeddings
        _, _, _, n_tokens, channels = input_sequence.shape

        # temporal embeddings
        if use_temporal_encoding:
            input_sequence = input_sequence + self.temporal_encoding
        # positional embeddings
        # if use_positional_encoding:
        #     input_sequence = input_sequence + self.positional_encoding
        # per camera embeddings
        if use_camera_encoding:
            input_sequence = input_sequence + self.camera_encoding

        embeds = input_sequence.view(BS, -1, channels)

        return embeds, (num_frames, n_tokens, channels)



if __name__ == "__main__":
    model = LLaVAnextEncoderModel(
        variant="llava-hf/llava-v1.6-mistral-7b-hf",
        debug=True,
        embed_dim=256,
        freeze=True,
    )
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

    # BS, N_FRAMES, N_CAMS, 3, H, W
    # random vector between 0 and 1
    image = np.random.rand(2, 4, 2, 3, 512, 1024).astype(np.float32)
    image = torch.tensor(image)

    # merge BS and N_FRAMES and N_CAMS
    image = image.view(-1, 3, 512, 1024)

    inputs = processor.image_processor(image, return_tensors="pt").to("cuda:0")
     # typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List[ForwardRef('PIL.Image.Image')], typing.List[numpy.ndarray], typing.List[ForwardRef('torch.Tensor')]]

    output = model(**inputs)
    print(output[0].shape, output[1])

    # output_image_shape = True
    # output = model(image, output_image_shape)
    # print(output.shape)