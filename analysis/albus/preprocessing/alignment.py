# Find translation between baseline images of different runs
# and store to file.

import json
from pathlib import Path

import cv2
import darsia
import matplotlib.pyplot as plt
import numpy as np
import skimage

# Assume the first image in each folder for each run can be used as
# reference image for this run.

with open("user_data_alignment.json", "r") as openfile:
    user_data = json.load(openfile)
main_path = user_data["albus folder"]

# TODO include AC01
# ac01 = main_path / Path("AC01")
ac02 = main_path / Path("AC02")
ac03 = main_path / Path("AC03")
ac04 = main_path / Path("AC04")
ac05 = main_path / Path("AC05")
ac06 = main_path / Path("AC06")
ac07 = main_path / Path("AC07")
ac08 = main_path / Path("AC08")
ac09 = main_path / Path("AC09")
ac10 = main_path / Path("AC10")
ac14 = main_path / Path("AC14")
ac19 = main_path / Path("AC19")
ac22 = main_path / Path("AC22")


base = {
    #    "ac01": list(sorted(ac01.glob("*")))[0],
    "ac02": list(sorted(ac02.glob("*")))[0],
    "ac03": list(sorted(ac03.glob("*")))[0],
    "ac04": list(sorted(ac04.glob("*")))[0],
    "ac05": list(sorted(ac05.glob("*")))[0],
    "ac06": list(sorted(ac06.glob("*")))[0],
    "ac07": list(sorted(ac07.glob("*")))[0],
    "ac08": list(sorted(ac08.glob("*")))[0],
    "ac09": list(sorted(ac09.glob("*")))[0],
    "ac10": list(sorted(ac10.glob("*")))[0],
    "ac14": list(sorted(ac14.glob("*")))[0],
    "ac19": list(sorted(ac19.glob("*")))[0],
    "ac22": list(sorted(ac22.glob("*")))[0],
}

img = {}
for key in base.keys():
    img[key] = cv2.cvtColor(
        cv2.imread(str(base[key])),
        cv2.COLOR_BGR2RGB,
    )

# Find the translation for all to the reference base, which is AC01
ref_img = img["ac02"].copy()  # TODO update
roi = (slice(1370, 2600), slice(3800, 5200))

# plt.imshow(ref_img)
# plt.show()

# Define drift object measuring the distance
translation_estimator = darsia.TranslationEstimator()
translation = {}
for key in base.keys():
    (
        translation_result,
        translation_intact,
    ) = translation_estimator.find_effective_translation(img[key], ref_img, roi)
    assert translation_intact
    translation[key] = translation_result

cache_path = Path("../cache")
cache_path.mkdir(parents=True, exist_ok=True)

# Store translation to file
for key in base.keys():
    np.save(cache_path / Path(f"translation_{key}.npy"), translation[key])
