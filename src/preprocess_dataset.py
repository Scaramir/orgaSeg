import cv2
import numpy as np
import tifffile as tiff
# std
from os import listdir
from pathlib import Path
from argparse import ArgumentParser
# 3rd party
from PIL import Image
from pipe import filter
from scipy.signal import convolve2d


# ------------------- #
# SETUP CONFIGURATION #
# ------------------- #
# Parse arguments
parser = ArgumentParser(
    prog='preprocess_dataset.py',
    description='Preprocess dataset to make it more uniform.'
)
parser.add_argument("raw_img_path", help="Path to raw image folder.",
                    default="./../data/raw_data/raw_images", required=False)
parser.add_argument("raw_anno_path", help="Path to raw annotation folder containing JSON files.",
                    default="./../data/raw_data/annotations_json", required=False)
parser.add_argument("-o", "--out", help="Path to output folder.",
                    default="./../data/preprocessed", required=False)
parser.add_argument("-s", "--size", help="Size of output images.",
                    default=512, required=False)
parser.add_argument("-r", "--replace_vignette", help="Replace vignette with median color.",
                    default=False, required=False)

args = parser.parse_args()

RAW_IMG_PATH = Path(args.raw_img_path)
RAW_ANNO_PATH = Path(args.raw_anno_path)
OUTPUT_PATH = Path(args.out)

if not RAW_IMG_PATH.exists():
    raise FileNotFoundError(
        f"Path to raw images does not exist: {RAW_IMG_PATH}")

if not RAW_ANNO_PATH.exists():
    raise FileNotFoundError(
        f"Path to raw annotations does not exist: {RAW_ANNO_PATH}")

OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

# ------------------- #
# PREPROCESS DATASET  #
# ------------------- #

# Code to remove vignette from images

# kernel
kernel_size = 5
kernel = np.ones((kernel_size, kernel_size)) / kernel_size**2


def remove_vignette_inplace(img: np.ndarray) -> None:  # img shape (h, w)
    img_mask = img != 0
    img_mask = convolve2d(img_mask, kernel, mode='same')
    vignette_mask = img_mask < 0.5
    img[vignette_mask] = np.median(img[~vignette_mask])


# Get image filenames and filter them

temp_path = Path('./../data/preprocessed/temp')

for img_name in Path.glob(RAW_IMG_PATH, '*.jpg').stem:
    # convert jpg to tif
    img_stem = img_name.stem
    img = Image.open(img_name)
    img.save(img_name.replace('.jpg', '.tiff'))

for img_name in Path.glob(RAW_IMG_PATH, '*.tif*'):
    # if *small* in name, continue
    if ('small' in img_name.stem) or ('1_tiff_oaf.jpg' in img_name.stem):
        continue

    # if not, resize image
    img = Image.open(img_name)
    # detect vignette and replace with median color
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if args.replace_vignette:
        median = np.median(img)
        img[img == 0] = median
    
    img = Image.fromarray(img)
    img = img.resize((args.size, args.size))

    img.save(OUTPUT_PATH / "images" / img_name.stem.replace('.tiff', '_resized.tiff'))
