#!/bin/bash
cd preprocessing
python preprocessing.py
cd ../c1
python segmentation.py
cd ../c2
python segmentation.py
cd ../c3
python segmentation.py
cd ../c4
python segmentation.py
cd ../c5
python segmentation.py
