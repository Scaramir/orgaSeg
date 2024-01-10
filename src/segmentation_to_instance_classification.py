import sys, os
import tifffile
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from pathlib import Path
from argparse import ArgumentParser

# ------------------- #
# SETUP CONFIGURATION #
# ------------------- #
# Parse arguments
parser = ArgumentParser(
    prog='segmentation_to_instance_classification.py',
    description='Convert segmentation to instance classification. Train a model to segment instances beforehand.'
)
parser.add_argument("--raw_img_path", help="Path to raw images.", 
                    default="./../data/data_sets/stardist/val", required=False)
parser.add_argument("--pred_mask_path", help="Path to predicted masks.",
                    default="./../data/data_sets/stardist_out/predicted_masks", required=False)
parser.add_argument("--out", help="Path to output folder.",
                    default="./../data/data_sets/classification/segmented_instances", required=False)
args, _ = parser.parse_known_args()  # Ignore unexpected arguments


raw_img_path = Path(args.raw_img_path)
pred_mask_path = Path(args.pred_mask_path)
output_path = Path(args.out)

if not raw_img_path.exists():
    raise FileNotFoundError(
        f"Path to raw images does not exist: {raw_img_path}")

if not pred_mask_path.exists():
    raise FileNotFoundError(
        f"Path to predicted masks does not exist: {pred_mask_path}")

output_path.mkdir(exist_ok=True, parents=True)

# ------------------- #
#     EXTRACTRION     #
# ------------------- #
def extract_instances_from_mask(mask: np.ndarray) -> List[np.ndarray]:
    """Extracts all instances from a mask.

    Args:
        mask (np.ndarray): mask to extract instances from.

    Returns:
        List[np.ndarray]: list with cooridnates of the instances' bounding box.
    """
    instance_boxes = []
    for i in mask.unique():
        if i == 0:
            continue
        instance = np.argwhere(mask == i)
        (ystart, xstart), (ystop, xstop) = instance.min(0), instance.max(0) + 1
        instance_boxes.append(mask[ystart:ystop, xstart:xstop])
    return instance_boxes

def crop_ori_image_to_instance(instance: np.ndarray, ori_image: np.ndarray) -> np.ndarray:
    """Crops the original image to the instance's bounding box.

    Args:
        instance (np.ndarray): instance to crop to.
        ori_image (np.ndarray): original image to crop.

    Returns:
        np.ndarray: cropped original image.
    """
    (ystart, xstart), (ystop, xstop) = np.argwhere(instance > 0).min(0), np.argwhere(instance > 0).max(0) + 1
    return ori_image[ystart:ystop, xstart:xstop]

def do_it_for_all_images(raw_img_path: Path, pred_mask_path: Path, output_path: Path) -> None:
    """Extracts all instances from all masks and crops the original image to the instance's bounding box.

    Args:
        raw_img_path (Path): path to the folder containing the original images.
        pred_mask_path (Path): path to the folder containing the predicted masks.
        output_path (Path): path to the output folder.
    """
    for _, img_name in enumerate(tqdm(list(Path.glob(raw_img_path, '*.tiff')), desc="Extracting instances from masks")):
        img = tifffile.imread(img_name)
        mask = tifffile.imread(str(img_name).replace('.tiff', '.png'))
        instances = extract_instances_from_mask(mask)
        for i, instance in enumerate(instances):
            cropped_img = crop_ori_image_to_instance(instance, img)
            tifffile.imwrite(str(output_path / f'{img_name.stem}_{i}.tiff'), cropped_img)
    return

if __name__ == "__main__":
    do_it_for_all_images(raw_img_path, pred_mask_path, output_path)
    print("Done.")
