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
              --root 'S:\studium\beliks_exploiting_dl_seminar\ORGAnoids_Applied_DL_FU\data\data_sets\stardist' \
              --out 'S:\studium\beliks_exploiting_dl_seminar\ORGAnoids_Applied_DL_FU\data\data_sets\stardist_out' \
              --epochs 20 \
              --steps_per_epoch 50 \
              --mode 2D \
              --train_patch_size 800,800")
    return

def stardist_predict():
    os.system("conda activate album && \
              album run stardist_predict \
              --root 'S:\studium\beliks_exploiting_dl_seminar\ORGAnoids_Applied_DL_FU\data\data_sets\stardist' \
              --out 'S:\studium\beliks_exploiting_dl_seminar\ORGAnoids_Applied_DL_FU\data\data_sets\stardist_out' \
              --model 'S:\studium\beliks_exploiting_dl_seminar\ORGAnoids_Applied_DL_FU\data\data_sets\stardist_out\stardist_model' \
              --axes 'YX' \
              --n_tiles 1 \
              --overlap 0.1 \
              --normalize_input 'percentile'")
    return

