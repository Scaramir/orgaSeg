# std
from os import listdir
from pathlib import Path
from argparse import ArgumentParser
# 3rd party
from pipe import filter


# Parse arguments

parser = ArgumentParser(
    prog='preprocess_dataset.py',
    description='Preprocess dataset to make it more uniform.'
)
parser.add_argument("DATASET_PATH", help="Path to dataset folder.")
parser.add_argument("-o", "--out", help="Path to output folder.",
                    default=None, required=False)

args = parser.parse_args()

DATASET_PATH = Path(args.DATASET_PATH)
OUTPUT_PATH = Path(args.out or args.DATASET_PATH + '_preprocessed')

if not OUTPUT_PATH.exists():
    OUTPUT_PATH.mkdir()


# Get image filenames and filter them

img_filenames = listdir(DATASET_PATH)

print(img_filenames)  # TODO: remove, only for testing


filtered_img_filenames = list(img_filenames
                              | filter(lambda x: not x.endswith('.jpg')))


print(filtered_img_filenames)  # TODO: remove, only for testing


# Load images and preprocess them
