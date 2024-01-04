'''
Maximilian Otto, 2022, @scaramir
Split the data of different classes/folders into test, train and validation folders with the desired ratio
The datasets get random shuffled before splitting.   
'''

import shutil
import random
import glob
import sys, os
import numpy as np
from tqdm import tqdm

# the root-folder containing the subdirectories/classes/resolutions
#   We need to adapt this to an output path as well and make it a parameter 
pic_folder_path = "./../data/preprocessed" 

# classes/subdirectories to include in the data sets. 
#   We have only one class for segmentation
#   But we can use this for classification as well when we have 6 classes
#   Therefore, this script needs some adjustments
class_dirs_to_use = ["images", "masks"]

# set some ratios:
train_ratio = 0.80 #.7
test_ratio = 0.20  #.3
val_ratio = 0.0

# TODO: convert ^ to sys.argv parameters, adjust some things, test it

# Global parameters for the current session:
# (reproducability)
def set_seed(seed=1129142083):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    return

# Disable print
def block_print():
    sys.stdout = open(os.devnull, 'w')

# Restore print
def enable_print():
    sys.stdout = sys.__stdout__

for cls in tqdm(class_dirs_to_use, desc = "Copy train/test/val sets"):
    #block_print()
    if not os.path.exists(pic_folder_path +'/train' + cls):
        os.makedirs(pic_folder_path +'/train' + cls)
    if not os.path.exists(pic_folder_path + '/val' + cls):
        os.makedirs(pic_folder_path +'/val' + cls)
    if not os.path.exists(pic_folder_path + '/test' + cls):
        os.makedirs(pic_folder_path +'/test' + cls)
    # Creating partitions of the data after shuffeling
    src = pic_folder_path + cls # Folder to copy images from

    allFileNames = glob.glob(src+"/*.tiff")
    
    # set seed so you get the same splits of every folder and therefore matching images and masks
    set_seed()

    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                              [int(len(allFileNames) * (1 - (val_ratio + test_ratio))), 
                                                               int(len(allFileNames) * (1 - test_ratio))])

    train_FileNames = [name for name in train_FileNames.tolist()]
    val_FileNames = [name for name in val_FileNames.tolist()]
    test_FileNames = [name for name in test_FileNames.tolist()]

    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, pic_folder_path +'/train' + cls)

    for name in test_FileNames:
        shutil.copy(name, pic_folder_path +'/test' + cls)

    for name in val_FileNames:
        shutil.copy(name, pic_folder_path +'/val' + cls)

#enable_print()
        
print("Done splitting data into train/test/val sets!")