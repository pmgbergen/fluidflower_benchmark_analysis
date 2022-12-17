#!/bin/bash

# Do not forget to update user_data.json files in each folder,
# and specify the path to the images.

# Preprocessing - generate geometry segmentation
cd preprocessing
python preprocessing.py

# Run segmentation for all experimental runs
cd bc01
python segmentation.py

cd ../bc02
python segmentation.py
