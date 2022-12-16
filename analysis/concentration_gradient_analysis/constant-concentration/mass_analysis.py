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

from total_injected_mass_large_FF import (total_mass_co2_port1,
                                          total_mass_co2_port2)

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

# ! ---- Material properties.
dissolution_limit = 1.8  # kg / m**3

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

    # Choose a subregion
    subregions = {}
    subregions["all"] = None  # represents entire domain.
    subregions["boxA"] = np.array([[1.1, 0.6], [2.8, 0.0]])
    subregions["boxB"] = np.array([[0, 1.2], [1.1, 0.6]])

    # Create empty lists for plotting purposes
    time_vec = []
    total_mass_co2_vec = []
    total_mass_mobile_co2_vec = {}
    total_mass_dissolved_co2_vec = {}
    total_mass_dissolved_co2_esf_vec = {}
    for item in ["port1", "port2", "total"]:
        total_mass_mobile_co2_vec[item] = {}
        total_mass_dissolved_co2_vec[item] = {}
        total_mass_dissolved_co2_esf_vec[item] = {}
        for roi in subregions.keys():
            total_mass_mobile_co2_vec[item][roi] = []
            total_mass_dissolved_co2_vec[item][roi] = []
            total_mass_dissolved_co2_esf_vec[item][roi] = []
    density_dissolved_co2_vec = {}
    for item in ["port1", "port2"]:
        density_dissolved_co2_vec[item] = []

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
        decomposed_seg = {}
        decomposed_seg["port1"] = np.zeros_like(seg, dtype=seg.dtype)
        decomposed_seg["port1"][rest] = seg[rest]

        decomposed_seg["port2"] = np.zeros_like(seg, dtype=seg.dtype)
        decomposed_seg["port2"][top_right] = seg[top_right]

        # Restrict the lower plume to the ESF layer above.
        esf_label = 3  # hardcoded
        for item in ["port1", "port2"]:
            decomposed_seg[item + "_esf"] = np.zeros_like(
                decomposed_seg[item], dtype=decomposed_seg[item].dtype
            )
            decomposed_seg[item + "_esf"][labels == esf_label] = decomposed_seg[item][
                labels == esf_label
            ]

        # ! ---- Perform sparse mass analysis on the segmentation.

        # Main idea. Treat injection through port1 and port2 separately and
        # sum them up. The difference between the two injections is that
        # both injection plumes have constant but different CO2 concentrations.
        # Determine various masses (total CO2, mobile CO2, dissolved CO2) and
        # consider various rois (entire domain, boxA, boxB).

        # Determine total mass of co2 (injected through port 1 and port 2)
        total_mass_co2 = {}
        total_mass_co2["port1"] = total_mass_co2_port1(t)
        total_mass_co2["port2"] = total_mass_co2_port2(t)
        total_mass_co2["total"] = sum(
            [total_mass_co2[item] for item in ["port1", "port2"]]
        )

        # Compute total mass of free co2 based on segmentation
        total_mass_mobile_co2 = {}
        for item in ["port1", "port2", "total"]:
            total_mass_mobile_co2[item] = {}

        for item in ["port1", "port2"]:
            for roi, subregion in subregions.items():
                total_mass_mobile_co2[item][roi] = mass_analysis.free_co2_mass(
                    decomposed_seg[item], pressure(t), 2, roi=subregion
                )
        for roi in subregions.keys():
            total_mass_mobile_co2["total"][roi] = sum(
                [total_mass_mobile_co2[item][roi] for item in ["port1", "port2"]]
            )

        # Compute total mass of dissolved CO2 as the difference between total mass and mass of free CO2
        total_mass_dissolved_co2 = {}
        for item in ["port1", "port2", "total"]:
            total_mass_dissolved_co2[item] = {}
            total_mass_dissolved_co2[item]["all"] = (
                total_mass_co2[item] - total_mass_mobile_co2[item]["all"]
            )

        # Prepare for seal analysis.
        total_mass_dissolved_co2_esf = {}
        for item in ["port1", "port2", "total"]:
            total_mass_dissolved_co2_esf[item] = {}

        # Compute volume of dissolved CO2.
        volume_dissolved_co2 = {}
        for item in ["port1", "port2", "port1_esf", "port2_esf"]:
            volume_dissolved_co2[item] = {}
            for roi, subregion in subregions.items():
                volume_dissolved_co2[item][roi] = mass_analysis.volume(
                    decomposed_seg[item], 1, roi=subregion
                )

        # Compute dissolved co2 in subregions and seal.
        # Assume constant mass concentration allowing to
        # use simple volume fractions as scaling.
        concentration_dissolved_co2 = {}
        for item in ["port1", "port2"]:
            if volume_dissolved_co2[item]["all"] > 1e-9:

                # Determine density/mass concentration of dissolved CO2.
                concentration_dissolved_co2[item] = max(
                    dissolution_limit,
                    total_mass_dissolved_co2[item]["all"]
                    / volume_dissolved_co2[item]["all"],
                )
            else:
                concentration_dissolved_co2[item] = 0.0

        # Determine total mass of dissolved CO2.
        for roi in ["all", "boxA", "boxB"]:
            for item in ["port1", "port2"]:
                total_mass_dissolved_co2[item][roi] = (
                    concentration_dissolved_co2[item] * volume_dissolved_co2[item][roi]
                )
                total_mass_dissolved_co2_esf[item][roi] = (
                    concentration_dissolved_co2[item]
                    * volume_dissolved_co2[item + "_esf"][roi]
                )
            total_mass_dissolved_co2["total"][roi] = sum(
                [total_mass_dissolved_co2[item][roi] for item in ["port1", "port2"]]
            )
            total_mass_dissolved_co2_esf["total"][roi] = sum(
                [total_mass_dissolved_co2_esf[item][roi] for item in ["port1", "port2"]]
            )

        # ! ---- Collect results.
        total_mass_co2_vec.append(total_mass_co2["total"])

        for item in ["port1", "port2", "total"]:
            for roi in subregions.keys():
                total_mass_mobile_co2_vec[item][roi].append(
                    total_mass_mobile_co2[item][roi]
                )
                total_mass_dissolved_co2_vec[item][roi].append(
                    total_mass_dissolved_co2[item][roi]
                )
                total_mass_dissolved_co2_esf_vec[item][roi].append(
                    total_mass_dissolved_co2_esf[item][roi]
                )

        for item in ["port1", "port2"]:
            density_dissolved_co2_vec[item].append(concentration_dissolved_co2[item])

        # ! --- Dense representation for CO2 concentration in water
        dense_concentration_dissolved_co2 = np.zeros(
            base_segmentation.img.shape[:2], dtype=float
        )

        # Masks for dissolved and mobile CO2
        for item in ["port1", "port2"]:
            mask_dissolved_co2 = decomposed_seg[item] == 1
            mask_mobile_co2 = decomposed_seg[item] == 2

            # Dissolved CO2 (in kg / m**3 - convert mass) - assume constant concentration for each plume
            dense_concentration_dissolved_co2[mask_dissolved_co2] = (
                concentration_dissolved_co2[item] / 1000
            )

            # Mobile CO2
            dense_concentration_dissolved_co2[mask_mobile_co2] = dissolution_limit

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

    # ! ---- Collect all data in excel sheets

    for item in ["port1", "port2", "total"]:

        df = pd.DataFrame()
        df["Time_[min]"] = time_vec

        df["Total_CO2"] = total_mass_co2_vec

        df["Mobile_CO2_[g]"] = total_mass_mobile_co2_vec[item]["all"]
        df["Dissolved_CO2_[g]"] = total_mass_dissolved_co2_vec[item]["all"]
        df["Dissolved_CO2_esf_[g]"] = total_mass_dissolved_co2_esf_vec[item]["all"]

        for roi in ["boxA", "boxB"]:
            df[f"Mobile_CO2_{roi}_[g]"] = total_mass_mobile_co2_vec[item][roi]
            df[f"Dissolved_CO2_{roi}_[g]"] = total_mass_dissolved_co2_vec[item][roi]
            df[f"Dissolved_CO2_esf_{roi}_[g]"] = total_mass_dissolved_co2_esf_vec[item][
                roi
            ]

        if item in ["port1", "port2"]:
            df[f"Concentration_CO2_{item}"] = density_dissolved_co2_vec[item]

        if user == "benyamine":
            df.to_excel(directory[-3:-1] + f"_{item}.xlsx", index=None)
        elif user == "jakub":
            df.to_excel(str(Path(f"{run_id}_{item}.xlsx")), index=None)
