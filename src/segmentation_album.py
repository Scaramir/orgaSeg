import os, sys
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm

# check if a conda environment called "album" exists
# if not, create one
def install_album_catalog():
    if os.system("conda env list | grep album") != 0:
        os.system("conda create -n album album python=3.10 -c conda-forge")
        os.system("conda activate album && album add-catalog https://gitlab.com/album-app/catalogs/image-challenges-dev.git")
        os.system("conda activate album && album install stardist_train && album install stardist_predict")
    return

def stardist_train():
    os.system("conda activate album && album run 'stardist_train' \
              --root 'ORGAnoids_Applied_DL_FU\data\data_sets\stardist' \
              --out 'ORGAnoids_Applied_DL_FU\data\data_sets\stardist_out' \
              --epochs 20 \
              --steps_per_epoch 50 \
              --mode 2D \
              --train_patch_size 800,800")
    return

def stardist_predict():
    os.system("conda activate album && \
              album run stardist_predict \
              --root 'ORGAnoids_Applied_DL_FU\data\data_sets\stardist' \
              --out 'ORGAnoids_Applied_DL_FU\data\data_sets\stardist_out' \
              --model 'ORGAnoids_Applied_DL_FU\data\data_sets\stardist_out\stardist_model' \
              --axes 'YX' \
              --n_tiles 1 \
              --overlap 0.1 \
              --normalize_input 'percentile'")
    return

# TODO: 
# convert folder and file structure to nnUNet format
# Write a function to run nnUNet plan and preprocess
# Write a function to run nnUNet train
# Write a function to run nnUNet predict

# TODO: compare the results of stardist and nnUNet
# TODO: Discuss the usage of Cellpose in our scenario (https://www.sciencedirect.com/science/article/pii/S2667290122000420)