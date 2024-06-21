import tifffile
import numpy as np
from tqdm import tqdm
from typing import List
from pathlib import Path
from argparse import ArgumentParser
from numpy import unique

# ------------------- #
# SETUP CONFIGURATION #
# ------------------- #
# Parse arguments
parser = ArgumentParser(
    prog="segmentation_to_instance_classification.py",
    description="Convert segmentation to instance classification. Train a segmentation model beforehand and use it to predict images of a folder. Then this script is used to crop out the individual instances and save them in a new folder before passing them to your already trained classifier.",
)
parser.add_argument(
    "--raw_img_path",
    help="Path to the raw images.",
    default="./../data/data_sets/stardist/val/images",
    required=False,
)
parser.add_argument(
    "--pred_mask_path",
    help="Path to the predicted masks.",
    default="./../data/data_sets/stardist/val/masks",
    required=False,
)
parser.add_argument(
    "--out",
    help="Path to the output folder.",
    default="./../data/data_sets/classification/segmented_instances",
    required=False,
)
args, _ = parser.parse_known_args()  # Ignore unexpected arguments


raw_img_path = Path(args.raw_img_path)
pred_mask_path = Path(args.pred_mask_path)
output_path = Path(args.out)

if not raw_img_path.exists():
    raise FileNotFoundError(f"Path to raw images does not exist: {raw_img_path}")

if not pred_mask_path.exists():
    raise FileNotFoundError(f"Path to predicted masks does not exist: {pred_mask_path}")

output_path.mkdir(exist_ok=True, parents=True)


# ------------------- #
#     EXTRACTRION     #
# ------------------- #
def extract_instances_from_mask(mask: np.ndarray) -> list[list[int]]:
    """Extracts all instances from a mask.

    Args:
        mask (np.ndarray): mask to extract instances from.

    Returns:
        List[np.ndarray]: list with coordinates of the instances' bounding box.
    """
    instance_boxes = []
    for i in unique(mask):
        if i == 0:
            continue
        instance = np.argwhere(mask == i)
        top_left, bottom_right = instance.min(0), instance.max(0) + 1
        instance_boxes.append([top_left, bottom_right])
    return instance_boxes


def crop_ori_image_to_instance(
    instance: np.ndarray, ori_image: np.ndarray
) -> np.ndarray:
    """Crops the original image to the instance's bounding box.

    Args:
        instance (np.ndarray): instance to crop to.
        ori_image (np.ndarray): original image to crop.

    Returns:
        np.ndarray: cropped original image.
    """
    return ori_image[instance[0][0]: instance[1][0], instance[0][1]: instance[1][1]]


def do_it_for_all_images(raw_img_path: Path, pred_mask_path: Path, output_path: Path) -> None:
    """Extracts all instances from all masks and crops the original image to the instance's bounding box.

    Args:
        raw_img_path (Path): path to the folder containing the original images.
        pred_mask_path (Path): path to the folder containing the predicted masks.
        output_path (Path): path to the output folder.
    """
    mask_names = list(Path.glob(pred_mask_path, "*.tiff"))

    for i, img_name in enumerate(
        tqdm(
            list(Path.glob(raw_img_path, "*.tiff")),
            desc="Extracting instances from masks",
        )
    ):
        img = tifffile.imread(img_name)
        mask = tifffile.imread(mask_names[i])
        instances = extract_instances_from_mask(mask)
        for i, instance in enumerate(instances):
            cropped_img = crop_ori_image_to_instance(instance, img)
            tifffile.imwrite(
                str(output_path / f"{img_name.stem}_{i}.tiff"), cropped_img
            )
    return


if __name__ == "__main__":
    do_it_for_all_images(raw_img_path, pred_mask_path, output_path)
    print("Done.")
