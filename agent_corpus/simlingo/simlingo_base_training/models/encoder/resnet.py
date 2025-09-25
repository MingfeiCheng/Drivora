import numpy as np
import torch
from torch import nn
from transformers import AutoModel, LlavaNextProcessor

class ResnetEncoderModel(nn.Module):
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

        self.image_encoder = AutoModel.from_pretrained(variant)

        self.projection = nn.Linear(self.image_encoder.config.hidden_sizes[-1], embed_dim)
        # Embeddings: BS, N_FRAMES, N_CAMS, N_PATCHES, EMBED_DIM
        self.temporal_encoding = nn.Parameter(0.02 * torch.randn(1, self.num_frames, 1, 1, embed_dim))
        self.camera_encoding = nn.Parameter(0.02 * torch.randn(1, 1, self.num_cameras, 1, embed_dim))
        # self.positional_encoding = nn.Parameter(torch.zeros(1, 1, 1, tokens_per_frame, embed_dim)).to("cuda:0")
        
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

        BS, num_frames, num_cams, C, H, W = pixel_values.shape

        pixel_values = pixel_values.squeeze().view(1, C, H, W)


        img_embeddings = self.image_encoder(pixel_values=pixel_values).last_hidden_state # torch.Size([4, 512, 12, 32])
        img_embeddings = img_embeddings.flatten(2,3).transpose(1, 2) # torch.Size([4, 384, 512])

        img_embeddings = self.projection(img_embeddings)
        img_embeddings = img_embeddings.view(-1, num_frames, num_cams, img_embeddings.shape[-2], img_embeddings.shape[-1])

        input_sequence = img_embeddings
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
    model = ResnetEncoderModel(
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