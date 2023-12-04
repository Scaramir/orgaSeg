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
