# std
from os import listdir
from pathlib import Path
from argparse import ArgumentParser
from json import load
from typing import List, Tuple, Dict
# 3rd party
import numpy as np
from tifffile import imwrite
from PIL import Image, ImageDraw


# Constants

CLASSES = {"Filled in", "Early branching", "Cyst", "Burned out", "Ring", "Branched"}

# Parse arguments

parser = ArgumentParser(
    prog='convert_label',
    description='Convert QuPath project annotations to tiff maps.'
)
parser.add_argument("ANNOTATIONS_PATH", help="Path to exported annotations of QuPath.")
parser.add_argument("DATASET_PATH", help="Path to dataset folder.")
parser.add_argument("-o", "--out", help="Path to output folder.",
                    default=None, required=False)

args = parser.parse_args()

ANNOTATIONS_PATH = Path(args.ANNOTATIONS_PATH)  # Directory containing of QuPath project
DATASET_PATH = Path(args.DATASET_PATH)  # Directory containing images
OUTPUT_PATH = Path(args.out or args.ANNOTATIONS_PATH + '_converted')

if not OUTPUT_PATH.exists():
    OUTPUT_PATH.mkdir()


# Read annotations & images

def get_annotations(annotations_file: Path
                    ) -> Dict[str, List[List[Tuple[int, int]]]]:
    try:
        with open(annotations_file) as f:
            raw_annotations = load(f)['features']
        img_annotations: Dict[str, List[List[Tuple[int, int]]]] = {}
        for raw_annotation in raw_annotations:
            class_label = raw_annotation["properties"]["classification"]["name"]
            coordinates = raw_annotation["geometry"]["coordinates"][0]
            if class_label not in img_annotations:
                img_annotations[class_label] = []
            img_annotations[class_label].append(coordinates)
        return img_annotations
    except KeyError:
        print(f'No img_annotations in {annotations_file}')
        return {}


def get_img_of_annotation_file(annotation_file: str) -> Image.Image:
    image_file = DATASET_PATH / (annotation_file[:-8] + '.tiff')
    if not image_file.exists():
        image_file = DATASET_PATH / (annotation_file[:-8] + '.jpg')
    img = Image.open(image_file)
    return img  # np.array(img)


annotation_file_names = listdir(ANNOTATIONS_PATH)
annotations = {
    f: get_annotations(ANNOTATIONS_PATH / f)
    for f in annotation_file_names}
image_files: Dict[str, Image.Image] = {
    f: get_img_of_annotation_file(f)
    for f in annotation_file_names}


# Create maps

for annotation_file_name in annotation_file_names:
    img_annotations = annotations[annotation_file_name]
    img_size = image_files[annotation_file_name].size
    class_maps = {class_label: Image.new("L", img_size)
                  for class_label in CLASSES}
    for class_label, class_annotations in img_annotations.items():
        draw = ImageDraw.Draw(class_maps[class_label])
        step_size = 255 // len(class_annotations)
        for i, annotation_coordinates in enumerate(class_annotations):
            if len(annotation_coordinates) == 1:
                annotation_coordinates = annotation_coordinates[0]  # required for special cases
            draw.polygon(list(map(tuple, annotation_coordinates)),
                         fill=255 - i * step_size)

    annotations_img = np.array(list(class_maps.values()))
    annotations_img_name = f'{annotation_file_name}_{"-".join(class_maps.keys())}.tiff'
    imwrite(OUTPUT_PATH / annotations_img_name, annotations_img)
