import os
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import darsia as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
from benchmark.standardsetups.binary_mass_analysis import BinaryMassAnalysis
from benchmark.utils.misc import concentration_to_csv, read_time_from_path
from benchmark.utils.time_from_image_name import ImageTime
from scipy import interpolate
from skimage.measure import label, regionprops

from total_injected_mass_large_FF import total_mass

# ! ---- Choose user.

user = "jakub"  # Alternatives: "benyamine", "jakub"

# ! ---- Provide path to the segmentations of C1, ..., C5.

if user == "benyamine":
    Results_path = (
        "E:/Git/fluidflower_benchmark_analysis/analysis/Results/fine_segmentation/"
    )
    seg_folders = [Results_path + i + "/" for i in os.listdir(Results_path)]

elif user == "jakub":
    main_results_path = Path(
        "/media/jakub/Elements/Jakub/benchmark/results/large_rig/fixed-thresholds"
    )
    segmentation_folder = Path("npy_segmentation")
    seg_folders = [
        main_results_path / Path(f"c{i}") / segmentation_folder for i in range(1, 6)
    ]

# ! ----  Injection times for C1, ..., C5.

inj_start_times = [
    datetime(2021, 11, 24, 8, 31, 0),  # c1
    datetime(2021, 12, 4, 10, 1, 0),  # c2
    datetime(2021, 12, 14, 11, 20, 0),  # c3
    datetime(2021, 12, 24, 9, 0, 0),  # c4
    datetime(2022, 1, 4, 11, 0, 0),  # c5
]

# ! ---- Fetch depth map.

# Provide path to depths (It is provided for the large FF in the "depths-folder")
if user == "benyamine":
    meas_dir = "E:/Git/fluidflower_benchmark_analysis/analysis/depths/"
    x_meas = meas_dir + "x_measures.npy"
    y_meas = meas_dir + "y_measures.npy"
    d_meas = meas_dir + "depth_measures.npy"
elif user == "jakub":
    meas_dir = Path(
        "/home/jakub/src/fluidflower_benchmark_analysis/analysis/depths/large_rig"
    )
    x_meas = meas_dir / Path("x_measures.npy")
    y_meas = meas_dir / Path("y_measures.npy")
    d_meas = meas_dir / Path("depth_measures.npy")

depth_measurements = (
    np.load(x_meas),
    np.load(y_meas),
    np.load(d_meas),
)

# ! ---- Segmentation of the geometry providing a mask for the lower ESF layer.

if user == "benyamine":
    labels_path = "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_port1/labels_fine.npy"
elif user == "jakub":
    labels_path = Path(
        "/home/jakub/src/fluidflower_benchmark_analysis/analysis/large_rig/cache/labels_fine.npy"
    )

labels = np.load(labels_path)

# ! ---- Analyze each run separately.

for i, directory in enumerate(seg_folders):

    run_id = f"c{i+1}"
    print(f"Start with run {run_id}.")

    # ! ---- Measurements.

    # Fetch injection start
    inj_start = inj_start_times[i]

    # Fetch temporal pressure data
    if user == "benyamine":
        pressure_data = "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/Florida_2021-11-24_2022-02-02_1669032742.xlsx"
    elif user == "jakub":
        pressure_data = Path(
            "/home/jakub/src/fluidflower_benchmark_analysis/analysis/mass_analysis_new/Florida_2021-11-24_2022-02-02_1669032742.xlsx"
        )
    df = pd.read_excel(pressure_data)

    # Extract time and add to the pressure data.
    date = [
        datetime.strptime(df.Dato.loc[i] + " " + df.Tid.loc[i], "%Y-%m-%d %H:%M")
        for i in range(len(df))
    ]
    df["Date"] = date

    # Reduce to the relevant times.
    df = df[
        (df.Date >= inj_start - timedelta(minutes=10))
        & (df.Date <= inj_start + timedelta(days=5) + timedelta(minutes=10))
    ]

    # Add time increments in minutes
    df["dt"] = ((df.Date - inj_start).dt.total_seconds()) / 60

    # Extract pressure data
    df["Lufttrykk"] = df.Lufttrykk + 3.125  # adjust for the height difference

    # Interpolate to get a function for atmospheric pressure over time (and scale pressures to bar)
    # pressure = interpolate.interp1d(time_pressure, 0.001 * atmospheric_pressures)
    pressure = interpolate.interp1d(df.dt.values, 0.001 * df.Lufttrykk.values)

    # ! ---- Mass analysis object.

    # Fetch the baseline image
    if user == "benyamine":
        list_dir = [i for i in os.listdir(directory) if i.endswith(".npy")]
        base_path = os.path.join(directory, list_dir[0])
    elif user == "jakub":
        list_dir = list(sorted(directory.glob("*.npy")))
        base_path = list_dir[0]

    base_segmentation = da.Image(
        np.load(base_path),
        width=2.8,
        height=1.5,
        color_space="GRAY",
    )

    # Create mass analysis object (the porosity could also be provided
    # as an np.ndarray with varying porosities depending on the sand layers.)
    mass_analysis = BinaryMassAnalysis(
        base_segmentation,
        depth_measurements=depth_measurements,
        porosity=0.44,
        cache="./cache/depth.npy",
    )

    # ! ---- Data structures.

    # Create empty lists for plotting purposes
    time_vec = []
    total_mass_co2_vec = []
    total_mass_mobile_co2_vec = []
    total_mass_mobile_co2_subregion1_vec = []
    total_mass_mobile_co2_subregion2_vec = []
    total_mass_dissolved_co2_vec = []
    total_mass_dissolved_co2_subregion1_vec = []
    total_mass_dissolved_co2_subregion2_vec = []
    total_mass_dissolved_co2_esf_vec = []
    total_mass_dissolved_co2_esf_subregion1_vec = []
    total_mass_dissolved_co2_esf_subregion2_vec = []
    density_dissolved_co2 = []

    # Choose a subregion
    subregion1 = np.array([[1.1, 0.6], [2.8, 0.0]])  # boxA
    subregion2 = np.array([[0, 1.2], [1.1, 0.6]])  # boxB

    # Initialize relative time, corresponding to the injection start
    t = 0
    if user == "jakub":
        t_ref = read_time_from_path(base_path)

    # ! ---- Actual analysis for the specific run.

    # Loop through directory of segmentations (extract last images in which the plumes from the two injectors merge)
    for c, im in enumerate(list_dir[:-6]):
        print(c, im)

        # ! ---- Fetch data (time and segmentation)

        # Update relative time by reading the time from the file name and comparing with previous

        # TODO Does this if statement introduce a inconsistency? may want to use t_new
        if user == "benyamine":
            if c != 0:
                t += ImageTime.dt(list_dir[c - 1], list_dir[c])
            time_vec.append(t)
        elif user == "jakub":
            t_absolute = read_time_from_path(im)
            t = (t_absolute - t_ref).total_seconds() / 60
            time_vec.append(t)

        # Fetch segmentation
        if user == "benyamine":
            seg = np.load(os.path.join(directory, im))
        elif user == "jakub":
            seg = np.load(im)

        # ! ---- Decompose segmentation into regions corresponding to injection in port 1 and port 2.

        # Label the segmentation and determine the region properties.
        seg_label = label(seg)
        regions = regionprops(seg_label)

        # Define the top right and rest of the image
        top_right = np.zeros_like(seg, dtype=bool)
        rest = np.zeros_like(seg, dtype=bool)

        # Loop through regions and check if centroid is in top right - if so discard that region.
        for i in range(len(regions)):
            if (
                0 < regions[i].centroid[0] < 2670  # y coordinate
                and 3450 < regions[i].centroid[1] < seg.shape[1]  # x coordinate
            ):
                top_right[seg_label == regions[i].label] = True
            else:
                rest[seg_label == regions[i].label] = True

        # Decompose segmentation into the injections of first and seconds well.
        seg_port1 = np.zeros_like(seg, dtype=seg.dtype)
        seg_port1[rest] = seg[rest]

        seg_port2 = np.zeros_like(seg, dtype=seg.dtype)
        seg_port2[top_right] = seg[top_right]

        # Restrict the lower plume to the ESF layer above
        esf_label = 3  # hardcode # hardcode
        seg_port1_esf = np.zeros_like(seg_port1, dtype=seg_port1.dtype)
        seg_port1_esf[labels == esf_label] = seg_port1[labels == esf_label]

        # ! ---- Perform sparse mass analysis on the segmentation corresponding to port 1.

        # Determine total mass of co2 (injected through port 1)
        total_mass_co2 = total_mass(t)
        total_mass_co2_vec.append(total_mass_co2)

        # Compute total mass of free co2 based on segmentation
        total_mass_mobile_co2 = mass_analysis.free_co2_mass(seg_port1, pressure(t), 2)
        total_mass_mobile_co2_subregion1 = mass_analysis.free_co2_mass(
            seg_port1, pressure(t), roi=subregion1
        )
        total_mass_mobile_co2_subregion2 = mass_analysis.free_co2_mass(
            seg_port1, pressure(t), roi=subregion2
        )
        total_mass_mobile_co2_vec.append(total_mass_mobile_co2)
        total_mass_mobile_co2_subregion1_vec.append(total_mass_mobile_co2_subregion1)
        total_mass_mobile_co2_subregion2_vec.append(total_mass_mobile_co2_subregion2)

        # Compute total mass of dissolved CO2 as the difference between total mass and mass of free CO2
        total_mass_dissolved_co2 = total_mass_co2 - total_mass_mobile_co2
        total_mass_dissolved_co2_vec.append(total_mass_dissolved_co2)

        # Compute volume of dissolved CO2 in entire rig
        volume_dissolved = mass_analysis.volume(seg_port1, 1)

        # Compute dissolved co2 in subregions and seal.
        # Assume constant mass concentration allowing to
        # use simple volume fractions as scaling.
        if volume_dissolved > 1e-9:

            # Compute volume of dissolved CO2 in the subregions
            volume_dissolved_subregion1 = mass_analysis.volume(
                seg_port1, 1, roi=subregion1
            )
            volume_dissolved_subregion2 = mass_analysis.volume(
                seg_port1, 1, roi=subregion2
            )

            # Compute volume of dissolved CO2 in seal
            volume_dissolved_esf = mass_analysis.volume(seg_port1_esf, 1)

            # Compute volume of dissolved CO2 in seal in subregions (restrict segmentation to the ESF layer for this)
            volume_dissolved_esf_subregion1 = mass_analysis.volume(
                seg_port1_esf, 1, roi=subregion1
            )
            volume_dissolved_esf_subregion2 = mass_analysis.volume(
                seg_port1_esf, 1, roi=subregion2
            )

            # Determine density/mass concentration of dissolved CO2.
            concentration_dissolved_co2 = total_mass_dissolved_co2 / volume_dissolved

            # Determine total mass of dissolved CO2.
            total_mass_dissolved_co2_subregion1 = (
                concentration_dissolved_co2 * volume_dissolved_subregion1
            )
            total_mass_dissolved_co2_subregion2 = (
                concentration_dissolved_co2 * volume_dissolved_subregion2
            )
            total_mass_dissolved_co2_esf = (
                concentration_dissolved_co2 * volume_dissolved_esf
            )
            total_mass_dissolved_co2_esf_subregion1 = (
                concentration_dissolved_co2 * volume_dissolved_esf_subregion1
            )
            total_mass_dissolved_co2_esf_subregion2 = (
                concentration_dissolved_co2 * volume_dissolved_esf_subregion2
            )

        else:
            # For numerical stability (to not divide by 0):
            concentration_dissolved_co2 = 0.0
            total_mass_dissolved_co2_subregion1 = 0.0
            total_mass_dissolved_co2_subregion2 = 0.0
            total_mass_dissolved_co2_esf = 0.0
            total_mass_dissolved_co2_esf_subregion1 = 0.0
            total_mass_dissolved_co2_esf_subregion2 = 0.0

        # Collect results
        density_dissolved_co2.append(concentration_dissolved_co2)
        total_mass_dissolved_co2_subregion1_vec.append(
            total_mass_dissolved_co2_subregion1
        )
        total_mass_dissolved_co2_subregion2_vec.append(
            total_mass_dissolved_co2_subregion2
        )
        total_mass_dissolved_co2_esf_vec.append(total_mass_dissolved_co2_esf)
        total_mass_dissolved_co2_esf_subregion1_vec.append(
            total_mass_dissolved_co2_esf_subregion1
        )
        total_mass_dissolved_co2_esf_subregion2_vec.append(
            total_mass_dissolved_co2_esf_subregion2
        )

        # ! --- Dense representation for CO2 concentration in water
        dense_concentration_dissolved_co2 = np.zeros(
            base_segmentation.img.shape[:2], dtype=float
        )

        # Masks for dissolved and mobile CO2
        mask_dissolved_co2_port1 = seg_port1 == 1
        mask_mobile_co2_port1 = seg_port1 == 2

        # Dissolved CO2 (in kg / m**3 - convert mass)
        dense_concentration_dissolved_co2[mask_dissolved_co2_port1] = (
            concentration_dissolved_co2 / 1000
        )

        # Mobile CO2
        dissolution_limit = 1.8  # kg / m**3
        dense_concentration_dissolved_co2[mask_mobile_co2_port1] = dissolution_limit

        # Store to file
        if user == "jakub":

            stem = im.stem

            results_folder = Path(
                "/media/jakub/Elements/Jakub/benchmark/results/large_rig/fixed-thresholds/"
            )
            run_folder = Path(f"{run_id}")

            # Store numpy arrays
            filename_npy = stem.replace("_segmentation", "_concentration") + ".npy"
            npy_concentration_folder = Path("concentration_npy")
            full_filename_npy = (
                results_folder
                / run_folder
                / npy_concentration_folder
                / Path(filename_npy)
            )
            full_filename_npy.parents[0].mkdir(parents=True, exist_ok=True)
            np.save(full_filename_npy, dense_concentration_dissolved_co2)

            # Store jpg images
            filename_jpg = stem.replace("_segmentation", "_concentration") + ".jpg"
            jpg_concentration_folder = Path("concentration_jpg")
            full_filename_jpg = (
                results_folder
                / run_folder
                / jpg_concentration_folder
                / Path(filename_jpg)
            )
            full_filename_jpg.parents[0].mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(full_filename_jpg),
                skimage.img_as_ubyte(
                    np.clip(dense_concentration_dissolved_co2 / dissolution_limit, 0, 1)
                ),
                # skimage.img_as_ubyte(1 - dense_concentration_dissolved_co2 / dissolution_limit),
                [int(cv2.IMWRITE_JPEG_QUALITY), 100],
            )

            # Store as coarse csv files, corresponding to 1cm by 1cm cells.
            dense_concentration_dissolved_co2_coarse = cv2.resize(
                dense_concentration_dissolved_co2,
                (280, 150),
                interpolation=cv2.INTER_AREA,
            )
            filename_csv = stem.replace("_segmentation", "_concentration") + ".csv"
            csv_concentration_folder = Path("concentration_csv")
            full_filename_csv = (
                results_folder
                / run_folder
                / csv_concentration_folder
                / Path(filename_csv)
            )
            full_filename_csv.parents[0].mkdir(parents=True, exist_ok=True)
            concentration_to_csv(
                full_filename_csv,
                dense_concentration_dissolved_co2_coarse,
                im.name,
            )

    df = pd.DataFrame()
    df["Time_[min]"] = time_vec
    df["Mobile_CO2_[g]"] = total_mass_mobile_co2_vec
    df["Dissolved_CO2_[g]"] = total_mass_dissolved_co2_vec
    df["Mobile_CO2_boxA_[g]"] = total_mass_mobile_co2_subregion1_vec
    df["Dissolved_CO2_boxA_[g]"] = total_mass_dissolved_co2_subregion1_vec
    df["Dissolved_CO2_esf_boxA_[g]"] = total_mass_dissolved_co2_esf_subregion1_vec
    df["Mobile_CO2_boxB_[g]"] = total_mass_mobile_co2_subregion2_vec
    df["Dissolved_CO2_boxB_[g]"] = total_mass_dissolved_co2_subregion2_vec
    df["Dissolved_CO2_esf_boxB_[g]"] = total_mass_dissolved_co2_esf_subregion2_vec
    df["Dissolved_CO2_esf_[g]"] = total_mass_dissolved_co2_esf_vec
    if user == "benyamine":
        df.to_excel(directory[-3:-1] + "_port1.xlsx", index=None)
    elif user == "jakub":
        df.to_excel(str(Path(f"{run_id}_port1.xlsx")), index=None)
