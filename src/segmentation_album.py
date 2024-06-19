import os
import subprocess
from PIL import Image


# check if a conda environment called "album" exists
# if not, create one
def install_album_catalog():
    if os.system("conda env list | grep album") != 0:
        os.system("conda create -n album album python=3.10 -c conda-forge")
        os.system(
            "conda activate album && album add-catalog https://gitlab.com/album-app/catalogs/image-challenges-dev.git"
        )
        os.system(
            "conda activate album && album install stardist_train && album install stardist_predict"
        )
    return


install_album_catalog()


def run_command(command):
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    # Capture the live output
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            print(output.strip().decode())


def stardist_train():
    command = (
        "conda activate album && "
        "album run stardist_train "
        "--root ./../data/data_sets/stardist "
        "--out ./../data/data_sets/stardist_out "
        "--epochs 100 "
        "--steps_per_epoch 200 "  # num_samples / batch_size = steps_per_epoch
        "--mode 2D "
        "--train_batch_size 4 "
        "--train_patch_size 176,176 "  # roughly a quarter of our down-sampled images
        "--total_memory 24000 "
        "--train_sample_cache True "  # default is True
    )
    run_command(command)
    return


stardist_train()


def stardist_predict():
    command = (
        "conda activate album && "
        "album run stardist_predict "
        "--root './../data/data_sets/stardist' "
        "--out './../data/data_sets/stardist_out' "
        "--model './../data/data_sets/stardist_out/stardist_model' "
        "--axes 'YX' "
        "--n_tiles 1 "
        "--overlap 0.1 "
        "--normalize_input 'percentile'"
    )
    run_command(command)
    return


stardist_predict()


# NOTE: When training a segmentation model, it is recommended to have all images in the same size. Stardist requires the images to be in the same size.
# You can use this function to check the shapes of the images in a folder.
def check_shapes_of_images_in_folder(folder_path):
    shapes = []
    for file in os.listdir(folder_path):
        if file.endswith(".tif"):
            img = Image.open(os.path.join(folder_path, file))
            shapes.append(img.size)
    return list(set(shapes))
