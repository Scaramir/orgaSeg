# ORGAnoids_applied_DL_FU
Applied Deep Learning Seminar @FU Berlin. Segment organoids and classify them into 6 morphological groups 


# Project Plan
## List of shit to do

1. Daten in einheitliche Form bringen
   - Unabhängiges Skript, um es ein mal auszuführen:
     - [x] alle in TIFFs umwandeln
     - [x] alle in greyscale speichern, aka nur 1 Farbkanal
     - [ ] optional: Vignette durch Median des Bildes ersetzen?
     - Bilder sinnvoll umbenennen? Vergrößerungsfaktor im Namen behalten. Ansonsten durchnummerieren. Tabelle mit original namen zu neuem namen speichern.
     - Bilder Vierteln um mehr Bilder zu generien 
     - die abspeichern
     - "2891 GREEN P4 D3 aR_0003" und "[...]BR_0000" löschen, weil die nicht gelesen werden können #kaputt
     - [x] "_small" löschen, die waren von anfang an kleiner und im lossy JPEG
     - [x] "1_tiff_oaf.jpg" löschen, weil falsche Auflösung.
  - Dynamischer Dataloader:
     - to_tensor()
     - Random_crop(514)
     - Rotation
     - Spiegeln Horizontal/Vertikal
     - Normalisieren
2. Ordnerstruktur für nnUNet aufbauen
3. Config vom nnUNet in Python/Pytorch-UNet anwenden und da implementieren
4. Versch. netze vergleichen
5. KLassifizieruzng diskutieren, dann weiter machen


3. unet selber 


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
