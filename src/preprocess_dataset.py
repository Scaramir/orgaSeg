import cv2
import numpy as np
import tifffile as tiff
import os, sys
# std
from pathlib import Path
from argparse import ArgumentParser
# 3rd party
from PIL import Image
from pipe import filter
from scipy.signal import convolve2d
from typing import List
from pathlib import Path


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

temp_path = Path('./../data/preprocessed/temp')

# Kernel setup to remove vignette from images:
kernel_size = 5.

# ------------------- #
# PREPROCESS DATASET  #
# ------------------- #
def convert_all_jpg_to_tiff(raw_image_path: Path) -> None:
    """Adds a tiff version of all JPEGs in the folder.

    Args:
        raw_image_path (Path): path to the folder containing the JPEGs. 
    """
    for img_name in Path.glob(raw_image_path, '*.jpg').stem:
        img = Image.open(img_name)
        img.save(img_name.replace('.jpg', '.tiff'))
    return 

def four_crop(img: np.ndarray) -> List[np.ndarray]:
    """get quadrants of image by using crops of size h/2, w/2

    Args:
        img (np.ndarray): image to be cropped.

    Returns:
        List[np.ndarray]: list of quadrants, top left, top right, bottom left, bottom right
    """
    h, w = img.shape
    crops = [
        img[:h//2, :w//2],
        img[:h//2, w//2:],
        img[h//2:, :w//2],
        img[h//2:, w//2:]
    ]
    return crops

def remove_vignette_inplace(img: np.ndarray, kernel_size: float = 5.) -> None:  # img shape (h, w)
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size**2
    img_mask = img != 0
    img_mask = convolve2d(img_mask, kernel, mode='same')
    vignette_mask = img_mask < 0.5
    img[vignette_mask] = np.median(img[~vignette_mask])
    return


# ------------------- #
# STUFF TO EXECUTE    #
# ------------------- #
for img_name in Path.glob(RAW_IMG_PATH, '*.tif*'):
    # if *small* or other weird shit in name, continue
    if ('small' in img_name.stem) or ('1_tiff_oaf.jpg' in img_name.stem):
        continue

    img = Image.open(img_name)
    img = np.array(img)
    # convert to greyscale image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if args.replace_vignette:
        img = remove_vignette_inplace(img, kernel_size)

    # TODO:
    # apply four crop to image and mask 
    # save the four crops by extending the original file name with the crop index


    # TODO: save each crop with desired size²
    img = Image.fromarray(img)
    img = img.resize((args.size, args.size))
    img.save(OUTPUT_PATH / "images" / img_name.stem.replace('.tiff', '_resized.tiff'))

