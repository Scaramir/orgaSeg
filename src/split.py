"""
Maximilian Otto, 2022, @scaramir
Split the data of different classes/folders into test, train and validation folders with the desired ratio
The datasets get randomly shuffled before splitting.   
"""

import shutil
import random
import glob
import sys
import os
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from typing import List


# Parse arguments
parser = ArgumentParser(
    prog="split.py",
    description="Splits a dataset into a train/val/test format. This script expects one or more folders containing *.tiff* files.",
)
parser.add_argument(
    "--pic_folder_path",
    help="The root-folder containing the required folder, where each folder to be split should contain the images.",
    default="./../data/preprocessed/big_data_set",
    required=False,
)
parser.add_argument(
    "--output_folder_path",
    help="The output path to write the images to",
    default="./../data/data_sets/stardist/big_data_set",
    required=False,
)
parser.add_argument(
    "--class_dirs_to_use",
    help="Folder names to include in the data sets (classes or subdirectories to split into train/test/val sets)",
    default=["images", "masks"],
    nargs="*",
    required=False,
)
parser.add_argument(
    "--train_ratio",
    help="Ratio of data to be used for training between 0 and 1",
    default=0.8,
    required=False,
)
parser.add_argument(
    "--test_ratio",
    help="Ratio of data to be used for testing between 0 and 1",
    # default=0.2,
    default=0.2,
    required=False,
)
parser.add_argument(
    "--val_ratio",
    help="Ratio of data to be used for validating between 0 and 1. Hint: Stardist expects the test images to be in a folder called `val` instead of `test`.",
    default=0.0,
    required=False,
)
args, _ = parser.parse_known_args()


# the root-folder containing the subdirectories/classes/resolutions
#   We need to adapt this to an output path as well and make it a parameter
pic_folder_path = args.pic_folder_path

# classes/subdirectories to include in the data sets.
#   We have only one class for segmentation
#   But we can use this for classification as well when we have 6 classes
#   Therefore, this script needs some adjustments
class_dirs_to_use = args.class_dirs_to_use

# output path to write the images to
output_folder_path = args.output_folder_path

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

if not os.path.exists(pic_folder_path):
    raise FileNotFoundError(f"Path to raw images does not exist: {pic_folder_path}")

# set some ratios:
train_ratio = args.train_ratio
test_ratio = args.test_ratio
val_ratio = args.val_ratio


# Global parameters for the current session:
# (reproducibility)
def set_seed(seed=1129142083):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return


# Disable print
def block_print():
    sys.stdout = open(os.devnull, "w")


# Restore print
def enable_print():
    sys.stdout = sys.__stdout__


def copy_files_to_subdir(filenames: List[str], subdir: str) -> None:
    for filename in filenames:
        shutil.copy(filename, output_folder_path + "/" + subdir + "/" + cls)


for cls in tqdm(class_dirs_to_use, desc="Copy train/test/val sets"):
    # block_print()
    for sub_dir in ["train", "test", "val"]:
        os.makedirs(output_folder_path + "/" + sub_dir + "/" + cls, exist_ok=True)
    # Creating partitions of the data after shuffling
    src = pic_folder_path + "/" + cls  # Folder to copy images from

    allFileNames = glob.glob(src + "/*.tif*")

    # set seed so you get the same splits of every folder and therefore matching images and masks
    set_seed()

    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames, test_FileNames = np.split(
        np.array(allFileNames),
        [
            int(len(allFileNames) * (1 - (val_ratio + test_ratio))),
            int(len(allFileNames) * (1 - test_ratio)),
        ],
    )

    # Copy-pasting images
    copy_files_to_subdir(train_FileNames.tolist(), "train")
    copy_files_to_subdir(test_FileNames.tolist(), "test")
    copy_files_to_subdir(val_FileNames.tolist(), "val")

# enable_print()

print("Done splitting data into train/test/val sets!")
