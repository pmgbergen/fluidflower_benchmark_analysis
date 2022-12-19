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
from benchmark.utils.misc import (concentration_to_csv, read_time_from_path,
                                  sg_to_csv, sw_to_csv)
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

# ! ---- Fetch spatial material properties
if user == "benyamine":
    porosity = 0.44
    swi = np.zeros((4260, 7951), dtype=float)
elif user == "jakub":
    porosity = np.load(
        "/home/jakub/src/fluidflower_benchmark_analysis/analysis/depths/large_rig/porosity.npy"
    )
    swi = np.load(
        "/home/jakub/src/fluidflower_benchmark_analysis/analysis/depths/large_rig/swi.npy"
    )
    zero_swi = True
    if zero_swi:
        swi[:, :] = 0

    depth = np.load(
        "cache/depth.npy"  # NOTE: Moved from constant concentration analysis.
    )

    # porosity = cv2.resize(porosity, (7952, 4260), interpolation = cv2.INTER_NEAREST)
    # swi = cv2.resize(swi, (7952, 4260), interpolation = cv2.INTER_NEAREST)
    # depth = cv2.resize(depth, (7952, 4260), interpolation = cv2.INTER_NEAREST)

# ! ---- Build volumes

Ny, Nx = porosity.shape[:2]
dx = 2.8 / Nx
dy = 1.5 / Ny
volume = np.multiply(porosity, depth) * dx * dy

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
        porosity=porosity,
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
    density_dissolved_co2_vec = {}
    for item in ["port1", "port2", "total"]:
        total_mass_mobile_co2_vec[item] = {}
        total_mass_dissolved_co2_vec[item] = {}
        for roi in ["boxA", "boxB", "esf", "all"]:
            total_mass_mobile_co2_vec[item][roi] = []
            total_mass_dissolved_co2_vec[item][roi] = []
        density_dissolved_co2_vec[item] = []

    # Initialize relative time, corresponding to the injection start
    t = 0
    if user == "jakub":
        t_ref = read_time_from_path(base_path)

    # ! ---- Actual analysis for the specific run.

    # Loop through directory of segmentations (extract last images in which the plumes from the two injectors merge)
    for c, im in enumerate(list_dir):
        print(c, im)

        # ! ---- Fetch data (time and segmentation)

        # Update relative time by reading the time from the file name and comparing with previous

        if user == "benyamine":
            # TODO Does this if statement introduce a inconsistency? may want to use t_new
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

        # Fetch masks
        water_mask = seg == 0
        aq_mask = seg == 1
        gas_mask = seg == 2

        # ! ---- Determine saturations

        # Full water saturation aside of residual saturations in gas regions
        sw = np.ones((Ny, Nx), dtype=float)
        sw[seg == 2] = swi[seg == 2]

        # Complementary condition for gas saturation
        sg = 1 - sw

        # Saturation times vol
        sw_vol = {}
        sg_vol = {}
        sw_vol["total"] = np.multiply(sw, volume)
        sg_vol["total"] = np.multiply(sg, volume)

        # ! ---- Density map
        co2_g_density = mass_analysis.external_pressure_to_density_co2(pressure(t))

        # ! ---- Free CO2 mass density
        mobile_co2_mass_density = {}
        mobile_co2_mass_density["total"] = np.multiply(sg_vol["total"], co2_g_density)

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
                and 3450 < regions[i].centroid[1] < Nx  # x coordinate
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

        decomposed_aq = {}
        decomposed_gas = {}
        for item in ["port1", "port2"]:
            decomposed_aq[item] = decomposed_seg[item] == 1
            decomposed_gas[item] = decomposed_seg[item] == 2

        # Restrict co2_mass_density to port1 and port2
        for item in ["port1", "port2"]:
            sg_vol[item] = np.multiply(sg_vol["total"], decomposed_gas[item])
            mobile_co2_mass_density[item] = np.multiply(sg_vol[item], co2_g_density)

        # ! ---- Brief sparse mass analysis.

        # Determine total mass of co2 (injected through port 1 and port 2)
        total_mass_co2 = {}
        total_mass_co2["port1"] = total_mass_co2_port1(t)
        total_mass_co2["port2"] = total_mass_co2_port2(t)
        total_mass_co2["total"] = sum(
            [total_mass_co2[item] for item in ["port1", "port2"]]
        )

        total_mass_co2_g = {}
        for item in ["port1", "port2", "total"]:
            total_mass_co2_g[item] = np.sum(mobile_co2_mass_density[item])

        total_mass_co2_aq = {}
        for item in ["port1", "port2", "total"]:
            total_mass_co2_aq[item] = total_mass_co2[item] - total_mass_co2_g[item]

        total_volume_co2_aq = {}
        for item in ["port1", "port2"]:
            sw_vol[item] = np.multiply(sw_vol["total"], decomposed_aq[item])
            total_volume_co2_aq[item] = np.sum(sw_vol[item])
        total_volume_co2_aq["total"] = np.sum(sw_vol["port1"] + sw_vol["port2"])

        effective_density_co2_aq = {}
        for item in ["port1", "port2", "total"]:
            effective_density_co2_aq[item] = (
                0
                if total_volume_co2_aq[item] < 1e-9
                else total_mass_co2_aq[item] / total_volume_co2_aq[item]
            )

        # Build dense CO2(aq) mass
        concentration_co2_aq = np.zeros((Ny, Nx), dtype=float)

        # Pick dissolution limit in the gaseous area.
        concentration_co2_aq[seg == 2] = dissolution_limit

        # Pick efective concentrations in two plumes (in kg / m**3)
        for item in ["port1", "port2"]:
            concentration_co2_aq[decomposed_aq[item]] = (
                effective_density_co2_aq[item] / 1000
            )

        # Treat case when plumes merge separately.
        if c > len(list_dir) - 6:
            concentration_co2_aq[seg == 1] = effective_density_co2_aq["total"]

        # Build spatial mass density
        dissolved_co2_mass_density = {}
        for item in ["port1", "port2", "total"]:
            dissolved_co2_mass_density[item] = (
                np.multiply(sw_vol[item], concentration_co2_aq) * 1000
            )  # g / m**3

        # Total mass density
        co2_total_mass_density = {}
        for item in ["port1", "port2", "total"]:
            co2_total_mass_density[item] = (
                dissolved_co2_mass_density[item] + mobile_co2_mass_density[item]
            )

        # ! ---- Make roi analysis
        esf_label = 3  # hardcoded
        esf = labels == esf_label

        # boxA = np.array([[1.1, 0.6], [2.8, 0.0]])
        # boxB = np.array([[0, 1.2], [1.1, 0.6]])
        box_a = (slice(int((1.5 - 1.1) / 1.5 * Ny), Ny), slice(int(1.1 / 2.8 * Nx), Nx))
        box_b = (
            slice(int((1.5 - 1.2) / 1.5 * Ny), int((1.5 - 0.6) / 1.5 * Ny)),
            slice(0, int(1.1 / 2.8 * Nx)),
        )

        entire_domain = np.ones((Ny, Nx), dtype=bool)

        subregions = {"boxA": box_a, "boxB": box_b, "esf": esf, "all": entire_domain}

        # ! ---- Collect results.
        total_mass_co2_vec.append(np.sum(co2_total_mass_density["total"]))

        for item in ["port1", "port2", "total"]:
            for key, roi in subregions.items():
                total_mass_mobile_co2_vec[item][key].append(
                    np.sum(mobile_co2_mass_density[item][roi])
                )
                total_mass_dissolved_co2_vec[item][key].append(
                    np.sum(dissolved_co2_mass_density[item][roi])
                )

            density_dissolved_co2_vec[item].append(effective_density_co2_aq[item])

        # Store to file
        if user == "jakub":

            stem = im.stem

            results_folder = Path(
                "/media/jakub/Elements/Jakub/benchmark/results/large_rig/fixed-thresholds/"
            )
            run_folder = Path(f"{run_id}")

            # Concentration

            if False:
                # Store numpy arrays
                filename_npy = stem.replace("_segmentation", "_concentration") + ".npy"
                npy_concentration_folder = (
                    Path("concentration_npy")
                    if zero_swi
                    else Path("swi_concentration_npy")
                )
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
                jpg_concentration_folder = (
                    Path("concentration_jpg")
                    if zero_swi
                    else Path("swi_concentration_jpg")
                )
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
                        np.clip(
                            dense_concentration_dissolved_co2 / dissolution_limit, 0, 1
                        )
                    ),
                    # skimage.img_as_ubyte(1 - dense_concentration_dissolved_co2 / dissolution_limit),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
                )

                # Sw

                # Store numpy arrays
                filename_npy = stem.replace("_segmentation", "_sw") + ".npy"
                npy_sw_folder = Path("sw_npy") if zero_swi else Path("swi_sw_npy")
                full_filename_npy = (
                    results_folder / run_folder / npy_sw_folder / Path(filename_npy)
                )
                full_filename_npy.parents[0].mkdir(parents=True, exist_ok=True)
                np.save(full_filename_npy, sw)

                # Store jpg images
                filename_jpg = stem.replace("_segmentation", "_sw") + ".jpg"
                jpg_sw_folder = Path("sw_jpg") if zero_swi else Path("swi_sw_jpg")
                full_filename_jpg = (
                    results_folder / run_folder / jpg_sw_folder / Path(filename_jpg)
                )
                full_filename_jpg.parents[0].mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(full_filename_jpg),
                    skimage.img_as_ubyte(sw),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
                )

                # Sg

                # Store numpy arrays
                filename_npy = stem.replace("_segmentation", "_sg") + ".npy"
                npy_sg_folder = cth("sg_npy") if zero_swi else Path("swi_sg_npy")
                full_filename_npy = (
                    results_folder / run_folder / npy_sg_folder / Path(filename_npy)
                )
                full_filename_npy.parents[0].mkdir(parents=True, exist_ok=True)
                np.save(full_filename_npy, sg)

                # Store jpg images
                filename_jpg = stem.replace("_segmentation", "_sg") + ".jpg"
                jpg_sg_folder = Path("sg_jpg") if zero_swi else Path("swi_sg_jpg")
                full_filename_jpg = (
                    results_folder / run_folder / jpg_sg_folder / Path(filename_jpg)
                )
                full_filename_jpg.parents[0].mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(full_filename_jpg),
                    skimage.img_as_ubyte(sg),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
                )

            if True:

                scaling = 4260 * 7952 / 280 / 150

                # Create coarse versions of mobile co2 mass density without loss of mass
                coarse_mobile_co2_mass_density = {}
                for item in ["port1", "port2", "total"]:
                    coarse_mobile_co2_mass_density[item] = (
                        cv2.resize(
                            mobile_co2_mass_density[item],
                            (280, 150),
                            interpolation=cv2.INTER_AREA,
                        )
                        * scaling
                    )

                # Boolean resize (nearest) - not conservative
                coarse_decomposed_aq = {}
                for item in ["port1", "port2"]:
                    coarse_decomposed_aq[item] = skimage.img_as_bool(
                        cv2.resize(
                            skimage.img_as_ubyte(decomposed_aq[item]),
                            (280, 150),
                            interpolation=cv2.INTER_AREA,
                        )
                    )

                co2_coarse = skimage.img_as_bool(
                    cv2.resize(
                        skimage.img_as_ubyte(
                            seg >= 1,
                        ),
                        (280, 150),
                        interpolation=cv2.INTER_AREA,
                    )
                )
                co2_gas_coarse = skimage.img_as_bool(
                    cv2.resize(
                        skimage.img_as_ubyte(
                            seg == 2,
                        ),
                        (280, 150),
                        interpolation=cv2.INTER_AREA,
                    )
                )
                coarse_seg = np.zeros((150, 280), dtype=np.uint8)
                coarse_seg[co2_coarse] += 1
                coarse_seg[co2_gas_coarse] += 1

                coarse_volume = (
                    cv2.resize(volume, (280, 150), interpolation=cv2.INTER_AREA)
                    * scaling
                )

                sg_coarse = coarse_seg == 2
                sw_coarse = 1 - sg_coarse

                coarse_sw_vol = {}
                coarse_sw_vol["total"] = np.multiply(coarse_volume, sw_coarse)
                for item in ["port1", "port2"]:
                    coarse_sw_vol[item] = np.multiply(
                        coarse_sw_vol["total"], coarse_decomposed_aq[item]
                    )

                coarse_total_volume_co2_aq = {}
                coarse_total_volume_co2_aq["port1"] = np.sum(coarse_sw_vol["port1"])
                coarse_total_volume_co2_aq["port2"] = np.sum(coarse_sw_vol["port2"])
                coarse_total_volume_co2_aq["total"] = np.sum(
                    coarse_sw_vol["port1"] + coarse_sw_vol["port2"]
                )

                coarse_effective_density_co2_aq = {}
                for item in ["port1", "port2", "total"]:
                    coarse_effective_density_co2_aq[item] = (
                        0
                        if coarse_total_volume_co2_aq[item] < 1e-9
                        else total_mass_co2_aq[item] / coarse_total_volume_co2_aq[item]
                    )

                # Build dense CO2(aq) mass
                coarse_concentration_co2_aq = np.zeros((150, 280), dtype=float)

                # Pick dissolution limit in the gaseous area.
                coarse_concentration_co2_aq[coarse_seg == 2] = dissolution_limit

                # Pick efective concentrations in two plumes (in kg / m**3)
                for item in ["port1", "port2"]:
                    coarse_concentration_co2_aq[coarse_decomposed_aq[item]] = (
                        coarse_effective_density_co2_aq[item] / 1000
                    )

                # Treat case when plumes merge separately.
                if c > len(list_dir) - 6:
                    coarse_concentration_co2_aq[
                        coarse_seg == 1
                    ] = coarse_effective_density_co2_aq["total"]

                # Build spatial mass density
                coarse_dissolved_co2_mass_density = {}
                for item in ["port1", "port2", "total"]:
                    coarse_dissolved_co2_mass_density[item] = (
                        np.multiply(coarse_sw_vol[item], coarse_concentration_co2_aq)
                        * 1000
                    )  # in g therefore scale

                print(
                    np.sum(coarse_dissolved_co2_mass_density["port1"]),
                    total_mass_co2_aq["port1"],
                    np.sum(dissolved_co2_mass_density["port1"]),
                )
                print(
                    np.sum(coarse_dissolved_co2_mass_density["port2"]),
                    total_mass_co2_aq["port2"],
                )

                # Total mass density
                coarse_co2_total_mass_density = {}
                for item in ["port1", "port2", "total"]:
                    coarse_co2_total_mass_density[item] = (
                        coarse_dissolved_co2_mass_density[item]
                        + coarse_mobile_co2_mass_density[item]
                    )

                print(
                    "test",
                    np.sum(coarse_co2_total_mass_density["total"]),
                    np.sum(co2_total_mass_density["total"]),
                )
                print(
                    "test",
                    np.sum(coarse_mobile_co2_mass_density["total"]),
                    np.sum(mobile_co2_mass_density["total"]),
                )
                print(
                    "test",
                    np.sum(coarse_dissolved_co2_mass_density["total"]),
                    np.sum(dissolved_co2_mass_density["total"]),
                )
                if False:
                    plt.figure("fine")
                    plt.imshow(concentration_co2_aq)
                    plt.figure("c")
                    plt.imshow(coarse_concentration_co2_aq)
                    plt.show()

                # Store as coarse csv files, corresponding to 1cm by 1cm cells.
                filename_csv = stem.replace("_segmentation", "_concentration") + ".csv"
                csv_concentration_folder = (
                    Path("concentration_csv")
                    if zero_swi
                    else Path("swi_concentration_csv")
                )
                full_filename_csv = (
                    results_folder
                    / run_folder
                    / csv_concentration_folder
                    / Path(filename_csv)
                )
                full_filename_csv.parents[0].mkdir(parents=True, exist_ok=True)
                concentration_to_csv(
                    full_filename_csv,
                    coarse_concentration_co2_aq,
                    im.name,
                )

                # Store as coarse csv files, corresponding to 1cm by 1cm cells.

                filename_csv = stem.replace("_segmentation", "_sg") + ".csv"
                csv_sg_folder = Path("sg_csv") if zero_swi else Path("swi_sg_csv")
                full_filename_csv = (
                    results_folder / run_folder / csv_sg_folder / Path(filename_csv)
                )
                full_filename_csv.parents[0].mkdir(parents=True, exist_ok=True)
                sg_to_csv(
                    full_filename_csv,
                    sg_coarse,
                    im.name,
                )

                # Store as coarse csv files, corresponding to 1cm by 1cm cells.
                filename_csv = stem.replace("_segmentation", "_sw") + ".csv"
                csv_sw_folder = Path("sw_csv") if zero_swi else Path("swi_sw_csv")
                full_filename_csv = (
                    results_folder / run_folder / csv_sw_folder / Path(filename_csv)
                )
                full_filename_csv.parents[0].mkdir(parents=True, exist_ok=True)
                sw_to_csv(
                    full_filename_csv,
                    sw_coarse,
                    im.name,
                )

    # ! ---- Collect all data in excel sheets

    for item in ["port1", "port2", "total"]:

        df = pd.DataFrame()
        df["Time_[min]"] = time_vec

        df["Total_CO2"] = total_mass_co2_vec

        df["Mobile_CO2_[g]"] = total_mass_mobile_co2_vec[item]["all"]
        df["Dissolved_CO2_[g]"] = total_mass_dissolved_co2_vec[item]["all"]

        for roi in ["boxA", "boxB", "esf"]:
            df[f"Mobile_CO2_{roi}_[g]"] = total_mass_mobile_co2_vec[item][roi]
            df[f"Dissolved_CO2_{roi}_[g]"] = total_mass_dissolved_co2_vec[item][roi]

        if item in ["port1", "port2"]:
            df[f"Concentration_CO2_{item}"] = density_dissolved_co2_vec[item]

        if user == "benyamine":
            df.to_excel(directory[-3:-1] + f"_{item}.xlsx", index=None)
        elif user == "jakub":
            excel_path = (
                Path(f"new_results/{run_id}_{item}.xlsx")
                if zero_swi
                else Path(f"new_swi_results/{run_id}_{item}.xlsx")
            )
            excel_path.parents[0].mkdir(parents=True, exist_ok=True)
            df.to_excel(str(excel_path), index=None)
