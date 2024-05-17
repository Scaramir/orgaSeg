# std
from os import listdir
from pathlib import Path
from argparse import ArgumentParser
from json import load
from typing import List, Tuple, Dict

# 3rd party
import numpy as np
from tqdm import tqdm
from tifffile import imwrite
from PIL import Image, ImageDraw


# Constants
CLASSES = {
    "Filled in",
    "Filled-in", # apparently the annotator used two different labels for the same class
    "Early branching",
    "Cyst",
    "Burned out",
    "Ring",
    "Branched",
    "Unsure",
    "apoptotic", # apparently there is a new class we didn't know about. thanks for wasting hours
}

# Parse arguments
parser = ArgumentParser(
    prog="convert_label", description="Convert QuPath project annotations to tiff maps."
)
parser.add_argument(
    "-a",
    "--ANNOTATIONS_PATH",
    help="Path to exported annotations of QuPath.",
    default="../../data/Oranoids for AI-20240330T140242Z-002/Oranoids for AI/All 713 pictures Qupath annotated/export/geojsons",
    required=False,
)
parser.add_argument(
    "-d",
    "--DATASET_PATH",
    help="Path to image dataset folder.",
    default="../data/raw_data/raw_images/big_data_set",
    required=False,
)
parser.add_argument(
    "-o",
    "--out",
    help="Path to output folder for created masks.",
    default="../data/preprocessed/anno_to_mask/big_data_set",
    required=False,
)
args, _ = parser.parse_known_args()  # Ignore unexpected arguments

ANNOTATIONS_PATH = Path(args.ANNOTATIONS_PATH)  # Directory containing of QuPath project
DATASET_PATH = Path(args.DATASET_PATH)  # Directory containing images
OUTPUT_PATH = Path(args.out)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# Read annotations & images
def get_annotations(annotations_file: Path) -> Dict[str, List[List[Tuple[int, int]]]]:
    try:
        with open(annotations_file) as f:
            raw_annotations = load(f)["features"]
        img_annotations: Dict[str, List[List[Tuple[int, int]]]] = {}
        for raw_annotation in raw_annotations:
            if "classification" not in raw_annotation["properties"]:
                continue
            class_label = raw_annotation["properties"]["classification"]["name"]
            coordinates = raw_annotation["geometry"]["coordinates"][0]
            if class_label not in img_annotations:
                img_annotations[class_label] = []
            img_annotations[class_label].append(coordinates)
        return img_annotations
    except KeyError:
        print(f"No img_annotations in {annotations_file}")
        return {}


def get_size_of_annotation_file(annotation_file: str) -> Tuple[int, int]:
    possible_img_paths = [
        DATASET_PATH / (annotation_file[:-8] + ".tiff"),
        DATASET_PATH / (annotation_file[:-8] + ".jpg"),
    ]
    image_file = DATASET_PATH / (annotation_file[:-8])
    for possible_img_path in possible_img_paths:
        if possible_img_path.exists():
            image_file = possible_img_path
            break
    img = Image.open(image_file)
    return img.size


def export_tiff(img: List[any], basename: str, details: str) -> None:
    imwrite(
        OUTPUT_PATH / f"{Path(basename).stem}_{details}.tiff",
        np.array(img, dtype=np.uint8),
    )


annotation_file_names = list(
    filter(lambda f: f.endswith(".geojson"), listdir(ANNOTATIONS_PATH))
)
annotations = {f: get_annotations(ANNOTATIONS_PATH / f) for f in annotation_file_names}
image_sizes = {f: get_size_of_annotation_file(f) for f in annotation_file_names}


# Create maps
for annotation_file_name in tqdm(annotation_file_names, desc="Creating maps"):
    img_annotations = annotations[annotation_file_name]
    img_size = image_sizes[annotation_file_name]
    class_maps = {
        class_label.lower(): Image.new("L", img_size) for class_label in CLASSES
    }
    segmentation_map = Image.new("L", img_size)
    segmentation_draw = ImageDraw.Draw(segmentation_map)
    segmentation_step_size = 255 // max(sum(map(len, img_annotations.values())), 1)
    segmentation_fill = 255
    for class_label, class_annotations in img_annotations.items():
        if class_label.lower() == "unsure":
            continue
        draw = ImageDraw.Draw(class_maps[class_label.lower()])
        step_size = 255 // len(class_annotations)
        for i, annotation_coordinates in enumerate(class_annotations):
            if len(annotation_coordinates) == 1:
                annotation_coordinates = list(annotation_coordinates[0])
                # required for special cases
            # if the annotator just drew a line instead a polygon:
            if not isinstance(annotation_coordinates[0], list):
                continue

            try:
                draw.polygon(
                    list(map(tuple, annotation_coordinates)), fill=255 - i * step_size
                )
                segmentation_draw.polygon(
                    list(map(tuple, annotation_coordinates)), fill=segmentation_fill
                )

                segmentation_fill -= segmentation_step_size
            except ValueError:
                print(f"Error in {annotation_file_name}. Skipping an objekt, because less semi-pixels would have to be drawn when creating a mask.")


    export_tiff(segmentation_map, annotation_file_name, "segmentation")

print("Done!")
