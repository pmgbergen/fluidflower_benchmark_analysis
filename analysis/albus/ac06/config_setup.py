"""
5-step config setup applied to the mueseum setup.
"""
from pathlib import Path

import cv2
import darsia
import matplotlib.pyplot as plt
import skimage

# !----- 1. Step: Read curved image and initialize the config file

# Choose a image of your choice.
#img = Path("//klient.uib.no/FELLES/LAB-IT/IFT/Resfys/medium_FF_AB_data/AB_image_sets/AC05/211026_time134837_DSC23343.JPG")
images = Path("/home/jakub/images/ift/medium/albus/AC06").glob("*")
img = list(images)[0]

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

# plt.figure("Original image")
# plt.imshow(img)
# plt.show()

# Add grid to the original image
original_image_gridded = darsia.Image(img, width = 0.899, height= 0.492, color_space = "RGB").add_grid(dx = 0.1, dy = 0.1, color =(250,30,30))

plt.figure("Original image - Grid")
plt.imshow(original_image_gridded.img)
plt.show()

# !----- 1. Step: ROI of Color checker for drift correction
# Define two corner points of a rectangle which contains the color checker.
# And be generous; this does not need (and should not) be too accurate.
# Use (row,col), i.e., (y,x) format.
config["drift"]["roi"] = [
    [700, 4840],
    [14, 5280],
]

# !----- 2. Step: Color checker
# Find the coordinates for the four marks on the color checker. Starting from the
# brown tile and proceeding counter clock wise. Use (x,y) format.
config["color"]["roi"] = [
    [4879, 604],
    [5242, 605],
    [5250, 63],
    [4888, 64],
]

# !----- 3. Step: Curvature correction

# Initialize curvature correction
curvature_correction = darsia.CurvatureCorrection(img)

# Read coordinates of 4 points, defining a rectangular of known dimensions.
# Here, we choose a bounding box with corners on the laser grid.

fluidflower_width = 0.899
fluidflower_height = 0.492

# Define config file for applying a homography, which after all transforms
# and crops the image to a box with known aspect ratio. This step already
# corrects for bulging in the normal direction due to the curvature of the
# FluidFlower.
config["curvature"]["crop"] = {
    "pts_src": [
        [94, 75],
        [139, 2954],
        [5420, 2966],
        [5441, 49],
    ],
    # Specify the true dimensions of the reference points - known as they are
    # points on the laser grid
    "width": fluidflower_width,
    "height": fluidflower_height,
}

# Extract quad ROI
img = darsia.extract_quadrilateral_ROI(img, **config["curvature"]["crop"])

# Add gridd to cropped image
cropped_image_gridded = darsia.Image(img, width = 0.899, height= 0.492, color_space = "RGB").add_grid(dx = 0.1, dy = 0.1, color =(250,30,30))

plt.figure("Cropped image - Grid")
plt.imshow(cropped_image_gridded.img)
plt.show()

# !----- 3. Step: Straighten the laser grid lines by correcting for bulge

# Plot...
# plt.figure("Cropped image")
# plt.imshow(img)
# plt.show()

# If not satisfied with the cropped image (due to bulge effects, uncomment the
# following code and correct for bulge.

## ... and determine the parameters as described in the darsia-notes
## For this, require the dimensions of the image
# Ny, Nx = img.shape[:2]
#
## Read the coordinates of the two largest impressions in y-direction (approx. top and bottom center)
# left = 0
# right = 0
# top = 28
# bottom = Ny - 4450
# (
#    horizontal_bulge,
#    horizontal_bulge_center_offset,
#    vertical_bulge,
#    vertical_bulge_center_offset,
# ) = curvature_correction.compute_bulge(
#    img=img, left=left, right=right, top=top, bottom=bottom
# )
#
## Choose horizontal and vertical bulge such that all laser grid lines are bulged inwards
# config["curvature"]["bulge"] = {
#    "horizontal_bulge": horizontal_bulge,
#    "vertical_bulge": vertical_bulge,
#    "horizontal_center_offset": horizontal_bulge_center_offset,
#    "vertical_center_offset": vertical_bulge_center_offset,
# }
#
## Apply final curvature correction
# img = curvature_correction.simple_curvature_correction(img, **config["bulge"])

# !----- Summary of the config - copy and move to another file.
print(config)

# This config file can now be used to run the predefined correction routine
# for any image of a large FluidFlower.
