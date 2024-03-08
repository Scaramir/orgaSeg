# ORGAnoids_applied_DL_FU
Applied Deep Learning Seminar @FU Berlin. Segment organoids and classify them into 6 morphological groups 


# Project Plan
## List of shit to do

1. Daten in einheitliche Form bringen
   - Unabhängiges Skript, um es ein mal auszuführen:
     - [x] alle in TIFFs umwandeln
     - [x] alle in greyscale speichern, aka nur 1 Farbkanal
     - [x] optional: Vignette durch Median des Bildes ersetzen? 
       - [ ] -> Testen, on das so wirklich funktioniert, oder ob überhaupt notwendig
     - Bilder sinnvoll umbenennen? Vergrößerungsfaktor im Namen behalten. Ansonsten durchnummerieren. Tabelle mit original namen zu neuem namen speichern?
     - [x] Bilder Vierteln um mehr Bilder zu generien 
     - [x] die abspeichern
     - "2891 GREEN P4 D3 aR_0003" und "[...]BR_0000" löschen, weil die nicht gelesen werden können #kaputt
     - [x] "_small" ignorieren, die waren von anfang an kleiner und im lossy JPEG
     - [x] "1_tiff_oaf.jpg" ignorieren, weil falsche Auflösung.
  - Dynamischer Dataloader für classifier bauen:
     - to_tensor()
     - Random_crop(514)
     - Rotation
     - Spiegeln Horizontal/Vertikal
     - Normalisieren

2. Ordnerstruktur für nnUNet, stardist, ... aufbauen
3. Config vom nnUNet in Python/Pytorch-UNet anwenden und da implementieren
4. Versch. netze vergleichen
5. Die einzelnen Objekte aus QuPath ziehen mit Class labels
6. Classifier scripts bauen, trainieren, testen und vergleichen -> Max' Scripts umschreiben und anwenden
   1. Achtung: 6 Klassen, ungleich verteilt
   2. auch hier preprocessing nötig (train/test-splitten, normailisieren (MEAN&STD extrahieren), augmentieren, etc.)
7. Klassifizierung diskutieren
8. Pipeline bauen, die automatisiert aus den segmentation predictions die position der objekte extrahiert und dann die Klassifizierung auf dem korrespondierenden Bildausschnitt des original-Bildes durchführt. Das Ergebnis wäre eine End-to-End Klassifizierung der Objekte im Bild.
9.  Im Report vergleichen wir dann die verschiedenen Ergebnisse der 
   1.  Segmentierung mit IoU, Dice, etc. und die 
   2.  Klassifizierung mit Accuracy, Precision, etc. und diskutieren die Ergebnisse. Wichtig: Class-Imbalance-Problem hervorheben, weil wir nur 6 Klassen haben und die nicht gleichverteilt sind. -> Gewichtung der Klassen in der Loss-Funktion und in der Accuracy-Funktion. -> F1-Score als Metrik verwenden.
10. Datenqualität diskutieren


# Directory structure
## data/ directory

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

## src/ directory

- _convert_label.py_: converts the annotations from geojson to tiff format
- _preprocess_dataset.py_: preprocesses the data set (raw images and tiff labels) to the preprocessed data set

# Pipeline flow

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

- Add image-challaenges-catalog to your album installation.  
- stardist want the subfolders for train/test to be namesd "images" and "masks", not "labels" -> why do we switched to "labels" in the first place? -> manually rename the folders or change the code that creates the folders
- Use segmentation_album.py to train stardist model
- TODO: Extend segmentation_album.py to train another segmentation model for comparison 

## Train classifier

- Use nn_pytorch.py to train a classifier

# Pipelining 
- TODO: automate the following steps in a pipeline
- Use your segmentation model to predict segmentation masks for a whole directory with new data (album solution: stardist_predict) 
- Use segmentation_to_classification.py to extract the objects from the segmentation masks and store them in a new directory
- Use your classifier to predict the class of the objects (nn_pytorch_predict.py) and obtain a csv file with the predictions for each object
