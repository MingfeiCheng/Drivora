#!/bin/bash
set -e

pip install networkx
pip install scipy
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install pytorch-lightning==1.9.5

pip install -e "$(dirname "$0")"

cd "$(dirname "$0")"

repo="rethinklab/Bench2DriveZoo"
repo_dir="Bench2DriveZoo" 

git lfs install

if [ -d "$repo_dir" ]; then
    echo "Repo $repo_dir exists skipping clone."
else
    echo "Start cloning $repo"
    git clone https://huggingface.co/$repo
fi