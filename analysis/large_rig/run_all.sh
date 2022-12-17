#!/bin/bash

# Prepare images and perform a series of correction routines.
# TODO

# Preprocessing - generate geometry segmentetion
cd preprocessing
python preprocessing_coarse.py
python preprocessing_fine.py

# Run segmentation for all experimental runs C1-5
cd c1
python segmentation.py
cd ../c2
python segmentation.py
cd ../c3
python segmentation.py
cd ../c4
python segmentation.py
cd ../c5
python segmentation.py

# Perform post analysis
# TODO
