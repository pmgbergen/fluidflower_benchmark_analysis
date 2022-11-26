#!/bin/bash

# Do not forget to update user_data.json files in each folder,
# and specify the path to the images.

# Preprocessing - generate geometry segmentation, and determine the drift of all setups relative to AC02
cd preprocessing
python processing.py
python alignment.py

# Run segmentation for all experimental runs
cd ac02
python segmentation.py

cd ../ac05
python segmentation.py
