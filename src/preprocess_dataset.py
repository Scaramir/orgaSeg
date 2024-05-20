import cv2
import numpy as np
import warnings
# std
from pathlib import Path
from argparse import ArgumentParser
# 3rd party
from PIL import Image
from tqdm import tqdm
from pipe import filter
from typing import List
from pathlib import Path
from tifffile import imwrite
from scipy.signal import convolve2d


# ------------------- #
# SETUP CONFIGURATION #
# ------------------- #
# Parse arguments
parser = ArgumentParser(
    prog='preprocess_dataset.py',
    description='Preprocess dataset to make it more uniform.'
)
parser.add_argument("-raw_img_path", help="Path to raw image folder.",
                    default="./../data/raw_data/raw_images/first_data_set", required=False)
parser.add_argument("-raw_mask_path", help="Path to raw annotation folder containing JSON files.",
                    default="./../data/preprocessed/anno_to_mask/first_data_set", required=False)
parser.add_argument("-o", "--out", help="Path to output folder.",
                    default="./../data/preprocessed/first_data_set", required=False)
parser.add_argument("-s", "--size", help="Size of output images. 1 is normal size, 0.5 is half size, etc.", 
                    default = 0.25, required=False)
parser.add_argument("-b", "--bg_elimination", help="Replace background with median color.",
                    default=False, required=False)
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
Path(OUTPUT_PATH / "images").mkdir(exist_ok=True, parents=True)
Path(OUTPUT_PATH / "masks").mkdir(exist_ok=True, parents=True)


# ------------------- #
# PREPROCESS DATASET  #
# ------------------- #
def convert_all_jpg_to_tiff(raw_image_path: Path) -> None:
    """Adds a tiff version of all JPEGs in the folder.

    Args:
        raw_image_path (Path): path to the folder containing the JPEGs. 
    """
    for _, img_name in enumerate(tqdm(list(Path.glob(raw_image_path, '*.jpg')), desc="Converting JPEGs to TIFFs")):
        img = Image.open(img_name)
        img.save(str(img_name).replace('.jpg', '.tiff'))
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
# Establish consistency in image format
convert_all_jpg_to_tiff(RAW_IMG_PATH)

# Get all image and mask files
img_files = list(RAW_IMG_PATH.glob("*.tif*"))
mask_files = list(RAW_MASK_PATH.glob("*.tiff"))

# Check if files exist
if len(mask_files) == 0:
    raise FileNotFoundError(f"No mask files found in {RAW_MASK_PATH}")

if len(img_files) == 0:
    raise FileNotFoundError(f"No image files found in {RAW_IMG_PATH}")

if len(img_files) != len(mask_files):
    warnings.warn("Number of image files does not match number of mask files !")

# The hard work
i = 0
def save_resize_crops(args, OUTPUT_PATH, mask_files, i, img_name, img_crops, mask_crops):
    for j, (img_crop, mask_crop) in enumerate(zip(img_crops, mask_crops)):
        if args.size:
            # resize image and mask, without using compression algorithms
            img_crop = cv2.resize(img_crop, (0, 0), fx=args.size, fy=args.size, interpolation=cv2.INTER_NEAREST)
            mask_crop = cv2.resize(mask_crop, (0, 0), fx=args.size, fy=args.size, interpolation=cv2.INTER_NEAREST)
        imwrite(OUTPUT_PATH / "images" / img_name.name.replace('.tiff', f'_{j}.tiff'), img_crop)
        imwrite(OUTPUT_PATH / "masks" / mask_files[i].name.replace('.tiff', f'_{j}.tiff'), mask_crop)

for img_name in tqdm(img_files, desc="Preprocessing images"):
    # if *small* or other weird stuff in name, continue
    if ('small' in img_name.stem) or ('1_tiff_oaf.jpg' in img_name.stem):
        i += 1
        continue
    # More explicit error messages
    try:
        if img_name.stem not in mask_files[i].stem:
            # TODO: fix warning. 
            warnings.warn(f"Mask file not found for image {img_name.stem}. Probably no matching prefix due to wrong file namings.")
            continue  # don't increase i
    except IndexError as e:
        print(str(e), "More images than masks found", img_name, ".\nAborting preprocessing...")
        break

    img = np.array(Image.open(img_name))
    mask = np.array(Image.open(mask_files[i]))
    # convert to greyscale image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if args.replace_vignette:
        remove_vignette_inplace(img)

    if args.bg_elimination:
        img = np.where(mask == 0, np.median(img[mask != 0]), img)

    img_crops = four_crop(img)
    mask_crops = four_crop(mask)

    save_resize_crops(args, OUTPUT_PATH, mask_files, i, img_name, img_crops, mask_crops)
    i += 1

print("DONE!")
