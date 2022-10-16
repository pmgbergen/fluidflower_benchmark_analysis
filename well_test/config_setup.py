"""
5-step config setup applied to the mueseum setup.
"""
from pathlib import Path

import cv2
import daria
import matplotlib.pyplot as plt
import skimage

# !----- 1. Step: Read curved image and initialize the config file

# Choose a image of your choice.
folder = Path("/home/jakub/images/ift/benchmark/well_test/baseline")
images = list(sorted(folder.glob("*")))
img = images[0]

# Read image
img = cv2.imread(str(img))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Brighten up image for better control
img = skimage.exposure.adjust_gamma(img, gamma=0.8)

# All relevant config parameters will be stored in a dictionary collecting several configs.
# Initialize the config dict.
curvature_correction = daria.CurvatureCorrection(img)
config: dict() = {}

# !----- 2. Step: Bulge
plt.imshow(img)
plt.show()

# Choose horizontal and vertical bulge such that all laser grid lines are bulged inwards.
# In some cases it might be necessary to define offsets for the image center;
# the default is to use the numerical center.
config["init"] = {
    "horizontal_bulge": 1e-9,
    "vertical_bulge": 0.0,
}

# Apply bulge 'correction'
img = curvature_correction.simple_curvature_correction(img, **config["init"])

# !----- 3. Step: Bulge

# Read coordinates of 4 points, defining a rectangular of known dimensions.
# Here, we choose a bounding box with corners on the laser grid.
plt.imshow(img)
plt.show()

fluidflower_width = 2.745
fluidflower_height = 1.5
frame_width = 0.00

# Define config file for applying a homography, which after all transforms
# and crops the image to a box with known aspect ratio. This step already
# corrects for bulging in the normal direction due to the curvature of the
# FluidFlower.
config["crop"] = {
    "pts_src": [
        # Without initial horizontal bulge
        # [27, 21],
        # [46, 4409],
        # [7915, 4391],
        # [7913, 10],
        [30, 21],
        [47, 4409],
        [7915, 4393],
        [7913, 10],
    ],
    # Specify the true dimensions of the reference points - known as they are
    # points on the laser grid
    "width": fluidflower_width + 2 * frame_width,
    "height": fluidflower_height + 2 * frame_width,
}

# Extract quad ROI
img = daria.extract_quadrilateral_ROI(img, **config["crop"])

# !----- 3. Step: Straighten the laser grid lines by correcting for bulge

# Plot...
plt.imshow(img)
plt.show()

# ... and determine the parameters as described in the daria-notes
# For this, require the dimensions of the image
Ny, Nx = img.shape[:2]

# Read the coordinates of the two largest impressions in y-direction (approx. top and bottom center)
left = 0
right = 0
top = 131
bottom = Ny - 4213
(
    horizontal_bulge,
    horizontal_bulge_center_offset,
    vertical_bulge,
    vertical_bulge_center_offset,
) = curvature_correction.compute_bulge(
    img=img, left=left, right=right, top=top, bottom=bottom
)

# Choose horizontal and vertical bulge such that all laser grid lines are bulged inwards
config["bulge"] = {
    "horizontal_bulge": horizontal_bulge,
    "vertical_bulge": vertical_bulge,
    "horizontal_center_offset": horizontal_bulge_center_offset,
    "vertical_center_offset": vertical_bulge_center_offset,
}

# Apply final curvature correction
img = curvature_correction.simple_curvature_correction(img, **config["bulge"])

# !----- 4. Step: Correct for stretch

# Compare with a 'perfect' grid layed on top
# Determine coordinates of some point which is off
da_img = daria.Image(
    img, width=config["crop"]["width"], height=config["crop"]["height"]
).add_grid(dx=0.1, dy=0.1)
plt.imshow(da_img.img)
plt.show()

# Choose horizontal and vertical bulge such that all laser grid lines are bulged inwards
config["stretch"] = {
    "horizontal_stretch": -3.95466460890734e-09,  # From museum calculations. See daria_lab/museum for details on how to obtain it.
    "horizontal_center_offset": 0.0,
    "vertical_stretch": 0.0,
    "vertical_center_offset": 0.0,
}

# Apply final curvature correction
img = curvature_correction.simple_curvature_correction(img, **config["stretch"])

# !----- 6. Step: Validation - Compare with a 'perfect' grid layed on top
da_img = daria.Image(
    img, width=config["crop"]["width"], height=config["crop"]["height"]
).add_grid(dx=0.1, dy=0.1)
plt.imshow(da_img.img)
plt.show()

# !----- 7. Step: Color correction

# Need to define a coarse ROI which contains the color checker - use [y,x] pixel ordering
config["color"] = {
    "roi": (slice(0, 600), slice(0, 600)),
}

# !----- Summary of the config - copy and move to another file.
print(config)

# This config file can now be used to run the predefined correction routine
# for any image of a large FluidFlower.
