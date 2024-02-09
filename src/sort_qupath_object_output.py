"""
sort files from qupath export into folders depending on class / annotation label.
Afterwards you can use split.py and pass the 6 classes to it so create train&test split.
"""

from pathlib import Path
import argparse
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description='Sorts QuPath object output into folders by class. The class has to be denoted in the file name in brackets. Every file has to have brackets with a annotation/class name in it.')
    parser.add_argument('--root', type=str, help='Path to root directory',
                        default='./../../annotation - organoid pictures/export/ROIsTif')
    parser.add_argument('--target_folder', type=str, help='Path to target folder',
                        default= './../data/data_sets/classification/objects')
    args, _ = parser.parse_known_args()
    return args


def get_file_names(root: Path):
    file_names = [name for name in root.glob('*.tif')]
    return file_names

# sort file names by whatever is written in '()'
def sort_file_names(file_names):
    # TODO: List index oput of range -> file names 
    sorted_file_names = sorted(file_names, key=lambda x: x.name.split('(')[1].split(')')[0])
    return sorted_file_names

def create_folders(target_folder: Path, sorted_file_names):

    for file_name in tqdm(sorted_file_names, desc='Creating folders and sorting files'):
        folder_name = file_name.name.split('(')[1].split(')')[0]
        folder = target_folder / folder_name
        folder.mkdir(exist_ok=True, parents=True)
        file_name.replace(folder / file_name.name)

if __name__ == '__main__':
    args = get_args()
    root = Path(args.root).resolve()
    target_folder = Path(args.target_folder).resolve()
    file_names = get_file_names(root)
    sorted_file_names = sort_file_names(file_names)
    create_folders(target_folder, sorted_file_names)
