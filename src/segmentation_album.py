import os
import subprocess

# check if a conda environment called "album" exists
# if not, create one
def install_album_catalog():
    if os.system("conda env list | grep album") != 0:
        os.system("conda create -n album album python=3.10 -c conda-forge")
        os.system("conda activate album && album add-catalog https://gitlab.com/album-app/catalogs/image-challenges-dev.git")
        os.system("conda activate album && album install stardist_train && album install stardist_predict")
    return
# install_album_catalog()

def stardist_train():
    import subprocess
    command = (
        "conda activate album && "
        "album run stardist_train "
        "--root S:/studium/ORGAnoids_Applied_DL_FU/data/data_sets/stardist/first_data_set "
        "--out S:/studium/ORGAnoids_Applied_DL_FU/data/data_sets/stardist_out/first_data_set "
        "--epochs 100 "
        "--steps_per_epoch 200 "  # num_samples / batch_size = steps_per_epoch (first_data_set: 50 steps aka batch_size=5)
        "--mode 2D "
        "--train_batch_size 4 "
        "--train_patch_size 176,176 "  # roughly a quarter of our down-sampled images
        "--total_memory 24000 "
        # "--train_sample_cache False " # default is True
    )
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # Capture the live output
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            print(output.strip().decode())
    return
stardist_train()

def stardist_predict():
    os.system("conda activate album && album run stardist_predict,\
              --root 'S:/studium/ORGAnoids_Applied_DL_FU/data/data_sets/stardist',\
              --out 'S:/studium/ORGAnoids_Applied_DL_FU/data/data_sets/stardist_out',\
              --model 'S:/studium/ORGAnoids_Applied_DL_FU/data/data_sets/stardist_out/stardist_model',\
              --axes 'YX',\
              --n_tiles 1,\
              --overlap 0.1,\
              --normalize_input 'percentile'")
    return

# TODO: 
# convert folder and file structure to nnUNet format
# Write a function to run nnUNet plan and preprocess
# Write a function to run nnUNet train
# Write a function to run nnUNet predict

# TODO: compare the results of stardist and nnUNet
# TODO: Discuss the usage of Cellpose in our scenario (https://www.sciencedirect.com/science/article/pii/S2667290122000420)


# NOTE: ALL images containing "oaf" are in the wrong resolution, are duplicates and we shoudl just ignore them
def check_shapes_of_images_in_folder(folder_path):
    import os
    from PIL import Image
    shapes = []
    for file in os.listdir(folder_path):
        if file.endswith(".tif"):
            img = Image.open(os.path.join(folder_path, file))
            shapes.append(img.size)
            # if img.size != (504, 380):
                # print(file)
    return list(set(shapes))
