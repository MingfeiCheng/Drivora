#!/bin/bash

set -e

# NOTE: the following parts maybe changed according to your system environment
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDA_HOME=/usr/local/cuda-11
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

pip install ninja packaging

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