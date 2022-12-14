"""
5-step config setup applied to the mueseum setup.
"""
from pathlib import Path

import cv2
import darsia
import matplotlib.pyplot as plt
import skimage

# !----- 0. Step: Read curved image and initialize the config file

# Choose a image of your choice.
img = Path("original/211124_time082740_DSC00067.JPG")

# Read image
img = cv2.imread(str(img))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Brighten up image for better control
img = skimage.exposure.adjust_gamma(img, gamma=0.7)

# All relevant config parameters will be stored in a dictionary collecting several configs.
# Initialize the config dict.
config: dict() = {}
config["drift"] = {}
config["color"] = {}
config["curvature"] = {
    "use_cache": True,
}

if True:
    plt.figure("Original image")
    plt.imshow(img)
    plt.show()

# !----- 1. Step: ROI of Color checker for drift correction
# Define two corner points of a rectangle which contains the color checker.
# And be generous; this does not need (and should not) be too accurate.
# Use (row,col), i.e., (y,x) format.
config["drift"]["roi"] = [
    [180, 100],
    [594, 544],
]

# !----- 2. Step: Color checker
# Find the coordinates for the four marks on the color checker. Starting from the
# brown tile and proceeding counter clock wise. Use (x,y) format.
config["color"]["roi"] = [
    [376, 494],
    [560, 502],
    [562, 242],
    [380, 233],
]

# !----- 2. Step: Bulge

# Initialize curvature correction
curvature_correction = darsia.CurvatureCorrection(img)

# Choose horizontal and vertical bulge such that all laser grid lines are bulged inwards.
# In some cases it might be necessary to define offsets for the image center;
# the default is to use the numerical center.
config["init"] = {
    "horizontal_bulge": 5e-10,
    "vertical_bulge": 0.0,
}

# Apply bulge 'correction'
img = curvature_correction.simple_curvature_correction(img, **config["init"])

# !----- 3. Step: Bulge

# Read coordinates of 4 points, defining a rectangular of known dimensions.
# Here, we choose a bounding box with corners on the laser grid.

if False:
    plt.figure("prebulged image")
    plt.imshow(img)
    plt.show()

fluidflower_width = 2.8
fluidflower_height = 1.5
frame_width = 0.00

# Define config file for applying a homography, which after all transforms
# and crops the image to a box with known aspect ratio. This step already
# corrects for bulging in the normal direction due to the curvature of the
# FluidFlower.
config["crop"] = {
    "pts_src": [[27, 25], [48, 4410], [7920, 4392], [7915, 13]],
    # Specify the true dimensions of the reference points - known as they are
    # points on the laser grid
    "width": fluidflower_width + 2 * frame_width,
    "height": fluidflower_height + 2 * frame_width,
}

# Extract quad ROI
img = darsia.extract_quadrilateral_ROI(img, **config["crop"])

# !----- 3. Step: Straighten the laser grid lines by correcting for bulge

## Plot...
if False:
    plt.figure("cropped image")
    plt.imshow(img)
    plt.show()

# ... and determine the parameters as described in the darsia-notes
# For this, require the dimensions of the image
Ny, Nx = img.shape[:2]

# Read the coordinates of the two largest impressions in y-direction (approx. top and bottom center)
left = 0
right = 0
top = 127
bottom = Ny - 4129
(
    horizontal_bulge,
    horizontal_bulge_center_offset,
    vertical_bulge,
    vertical_bulge_center_offset,
) = curvature_correction.compute_bulge(
    img=img, left=left, right=right, top=top, bottom=bottom
)

print(vertical_bulge, vertical_bulge_center_offset)

# Choose horizontal and vertical bulge such that all laser grid lines are bulged inwards
config["bulge"] = {
    "horizontal_bulge": -0.0,
    "horizontal_center_offset": 0,
    "vertical_bulge": vertical_bulge,
    "vertical_center_offset": vertical_bulge_center_offset,
}

# Apply final curvature correction
img = curvature_correction.simple_curvature_correction(img, **config["bulge"])

# Plot...
if False:
    plt.figure("bulged image")
    plt.imshow(img)
    plt.show()

# !----- 4. Step: Correct for stretch

# Fetch stretch from previous studies based on images with laser grids
config["stretch"] = {
    "horizontal_stretch": -2e-9,
    "horizontal_center_offset": -274,
    "vertical_stretch": 3.2177888435429053e-09,
    "vertical_center_offset": 702,
}

# Apply final curvature correction
img = curvature_correction.simple_curvature_correction(img, **config["stretch"])

# !----- 5. Step: Validation - Compare with a 'perfect' grid layed on top
da_img = darsia.Image(
    img, width=config["crop"]["width"], height=config["crop"]["height"]
).add_grid(dx=0.1, dy=0.1)

if True:
    plt.figure("stretched image")
    plt.imshow(da_img.img)
    plt.show()

# !----- Summary of the config - copy and move to another file.
print(config)

# This config file can now be used to run the predefined correction routine
# for any image of a large FluidFlower.
