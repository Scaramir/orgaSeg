import cv2
import numpy as np
import tifffile as tiff
import os, sys
# std
from pathlib import Path
from argparse import ArgumentParser
# 3rd party
from PIL import Image
from tqdm import tqdm
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
parser.add_argument("-raw_img_path", help="Path to raw image folder.",
                    default="./../data/raw_data/raw_images", required=False)
parser.add_argument("-raw_mask_path", help="Path to raw annotation folder containing JSON files.",
                    default="./../data/preprocessed/anno_to_mask", required=False)
parser.add_argument("-o", "--out", help="Path to output folder.",
                    default="./../data/preprocessed", required=False)
parser.add_argument("-s", "--size", help="Size of output images.",
                    required=False)
parser.add_argument("-r", "--replace_vignette", help="Replace vignette with median color.",
                    default=False, required=False)
args, _ = parser.parse_known_args()  # Ignore unexpected arguments

RAW_IMG_PATH = Path(args.raw_img_path)
RAW_MASK_PATH = Path(args.raw_mask_path)
OUTPUT_PATH = Path(args.out)

if not RAW_IMG_PATH.exists():
    raise FileNotFoundError(
        f"Path to raw images does not exist: {RAW_IMG_PATH}")

if not RAW_MASK_PATH.exists():
    raise FileNotFoundError(
        f"Path to raw annotations does not exist: {RAW_MASK_PATH}")

OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

# Kernel setup to remove vignette from images:
kernel_size = 5

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

def remove_vignette_inplace(img: np.ndarray, kernel_size: int = 5) -> None:  # img shape (h, w)
    """Removes vignette from image by replacing the vignette with the median color of the image.

    Args:
        img (np.ndarray): image to remove vignette from.
        kernel_size (int, optional): size of kernel to use for convolution. Defaults to 5.
    """
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size**2
    img_mask = img != 0
    img_mask = convolve2d(img_mask, kernel, mode='same')
    vignette_mask = img_mask < 0.5
    img[vignette_mask] = np.median(img[~vignette_mask])
    return


# ------------------- #
# STUFF TO EXECUTE    #
# ------------------- #
mask_files = list(RAW_MASK_PATH.glob("*.tiff"))
if len(mask_files) == 0:
    raise FileNotFoundError(f"No mask files found in {RAW_MASK_PATH}")

for i, img_name in enumerate(tqdm(Path.glob(RAW_IMG_PATH, '*.tif*'), desc="Preprocessing images")):
    # if *small* or other weird stuff in name, continue
    if ('small' in img_name.stem) or ('1_tiff_oaf.jpg' in img_name.stem):
        continue

    if img_name not in mask_files[i]:
        raise FileNotFoundError(f"Mask file not found for image {img_name}. Probably wrong order of files or no matching prefix.")

    img = Image.open(img_name)
    mask = Image.open(mask_files[i])
    # convert to greyscale image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if args.replace_vignette:
        img = remove_vignette_inplace(img, kernel_size)

    img_crops = four_crop(img)
    mask_crops = four_crop(mask)

    for j, (img_crop, mask_crop) in enumerate(zip(img_crops, mask_crops)):
        img_crop = Image.fromarray(img_crop)
        mask_crop = Image.fromarray(mask_crop)
        if args.size:
            img_crop = img_crop.resize((args.size, args.size))
            mask_crop = mask_crop.resize((args.size, args.size))
        img_crop.save(OUTPUT_PATH / "images" / img_name.stem.replace('.tiff', f'_{j}.tiff'))
        mask_crop.save(OUTPUT_PATH / "labels" / mask_files[i].stem.replace('.tiff', f'_{j}.tiff'))
    # TODO: test if this works and fix if not (probably not) ! 

print("DONE!")