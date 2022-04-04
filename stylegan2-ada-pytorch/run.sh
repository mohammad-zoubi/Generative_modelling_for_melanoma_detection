#!/usr/bin/env bash
sudo apt-get update
sudo apt install -y python3 python3-pip

sudo pip3 install mmcv-full==1.3.15 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
sudo pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
sudo pip3 install click
sudo pip3 install pillow
sudo pip3 install numpy
sudo pip3 install scipy
sudo pip3 install torch
sudo pip3 install ninja
sudo apt-get install ffmpeg libsm6 libxext6  -y
sudo pip3 install tqdm
sudo pip3 install wandb
sudo pip3 install efficientnet-pytorch
sudo pip3 install sklearn
sudo pip3 install pandas
sudo pip3 install moviepy
sudo pip3 install albumentations
sudo pip3 install tensorboard
sudo pip3 install matplotlib
sudo pip3 install seaborn