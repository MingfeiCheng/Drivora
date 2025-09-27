#!/bin/bash
set -e  

export CUDA_HOME=/usr/local/cuda-11 # can be changed according to your system environment

pip install setuptools==60.2.0
pip install networkx==2.5.1
pip install scipy==1.6.2
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install openmim
mim install mmcv-full==1.4.1
pip install mmdet===2.19.1
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install filterpy==1.4.5
pip install munkres==1.1.4
pip install rdp==0.8
pip install timm==0.4.12
pip install ujson==4.2.0
pip install scikit-image==0.18.1
pip install beartype==0.9.1
pip install einops==0.4.0
pip install pytorch-lightning==1.5.10
pip install safetensors==0.3.1
pip install transformers==4.30.2
pip install huggingface_hub==0.14.1
pip install torchmetrics==0.7.2

pip install -e "$(dirname "$0")"

# cd "$(dirname "$0")"

# chmod +x download.sh
# ./download.sh
