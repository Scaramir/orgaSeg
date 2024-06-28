# ORGAnoids_applied_DL_FU
Applied Deep Learning Seminar @FU Berlin. The purpose iof this project is to segment organoids and classify them into morphological groups.  
Due to the nature of the data, the project is divided into two parts: segmentation and classification.  
To obtain ground-truth data to train the supervised segmentation and classification models, an annotation using QuPath is expected.  
Alternatively, data folders with individual images, their crops containing one object at a time with a file name containing  `(class_name)` and the masks for the whole images can be provided as well. 






# Training Pipeline

![training steps](training.png)

## Export data

- Open project with annotations in QuPath
- Use src/export_geojsons_and_rois.groovy file with instructions on top to export cells & geojsons
- use sort_qupath_object_output.py to sort the exported data into the correct directory structure for training a classification model 

## Preprocess data

- Use convert_label.py to convert geojson files to tiffs
- Use preprocess_dataset.py to preprocess images and masks
- Use split.py to split segmentation data in train and test data, and create correct dataset structure for segmentation models
- Use split.py to split classification data in train and test data, and create correct dataset structure for classification models 


## Train segmentation

- Add image-challaenges-catalog to your album.solutions installation.  
- stardist wants the subfolders for train/test to be namesd "images" and "masks", not "labels"
- Use segmentation_album.py to train stardist model


## Train classifier

- Use nn_pytorch.py to train a classifier and store the model in the models directory

# Inference Pipeline 

![inference steps](inference.png)

- Use your segmentation model to predict segmentation masks for a whole directory with new data (album solution: stardist_predict) 
- Use segmentation_to_classification.py to extract the objects from the segmentation masks and store them in a new directory
- Use your trained classifier to predict the class of the objects (nn_pytorch.py) and obtain a csv file with the predictions for each object



# Directory structure
## `data/` directory

- _data_sets_: contains the preprocessed and model-specific reordered data sets used for training and testing
    - _..._: one subdirectory for each model
      - _test_: contains the test data set
        - _img_: contains the test images
          - as tiff files
        - _mask_: contains the test labels
          - as tiff files
      - _train_: contains the train data set
        - _img_: contains the train images
          - as tiff files
        - _mask_: contains the train labels
          - as tiff files
- _preprocessed_: contains the preprocessed data sets
    - _anno_to_mask_: contains the masks generated from the annotations (using the src/convert_label.py script)
      - as tiff files
    - _images_: contains the preprocessed images (using the src/preprocess_dataset.py script)
      - as tiff files
    - _labels_: contains the labels preprocessed from anno_to_mask (using the src/preprocess_dataset.py script)
      - as tiff files
- _raw_data_: contains the raw data sets (unchanged)
  - _annotations_json_: contains the annotations in geojson format
    - as geojson files
  - _raw_images_: contains the raw images
    - as jpg & tiff files

## `src/` directory

- _convert_label.py_: converts the annotations from geojson to tiff format
- _preprocess_dataset.py_: preprocesses the data set (raw images and tiff labels) to the preprocessed data set
