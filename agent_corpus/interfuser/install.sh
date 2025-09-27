#!/bin/bash
set -e  

pip install --upgrade setuptools
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install gdown

cd "$(dirname "$0")"

FILE_ID="1GKiASmGPbD4FwHkUoVfGRk_lMLrGb2f6"
FILE_NAME="interfuser.pth.tar"

gdown -c --id "$FILE_ID" -O "$FILE_NAME"

pip install -r requirements.txt

cd interfuser
python setup.py develop
