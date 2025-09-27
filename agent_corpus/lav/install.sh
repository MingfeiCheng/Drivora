#!/bin/bash
set -e  

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter==2.0.7 -f https://data.pyg.org/whl/torch-1.7.1+cu110.html
pip install einops==0.3.2
pip install gdown

cd "$(dirname "$0")"

FILE_ID="1xtG_m_freoR2wzRShd6dOJFA0dq--2iu"
FILE_NAME="weight.zip"


gdown -c --id "$FILE_ID" -O "$FILE_NAME"

unzip -o "$FILE_NAME" -d .

find . -type d -name "__MACOSX" -exec rm -rf {} +

rm -f "$FILE_NAME"