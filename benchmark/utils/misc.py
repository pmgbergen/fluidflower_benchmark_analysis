"""
Module collecting simple and general utilities.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Union


def read_time_from_path(path: Union[Path, str]) -> datetime:
    """
    Method to read hardcoded datetime path to datetime format.

    The conversion expects a very specific style for file name.
    They need to start with the date in yyMMdd format, following
    by an underscore, the keyword 'time' and then the time of the
    day in the format HHmmss. An example name is for instance:

    "folder/subfolder/211124_time083942_DSC00139.JPG"

    This method returns the datetime corresponding to 24. Nov 2021,
    8 hours, 39 minutes, 42 seconds.

    Args:
        path (Path or str): path to file; both full path and path without
            folders are supported.

    Returns:
        datetime.datetime: time
    """
    # Convert to Path
    if isinstance(path, str):
        path = Path(path)

    # Clean path from all folders
    path_file = path.name

    # Identify the first 6 letters as date in the format yyMMdd
    date: str = path_file[:6]

    # Identify letters 12-17 (starting to count from 1) as time in the format HHmmss
    time: str = path_file[11:17]

    # Convert to datetime
    return datetime.strptime(date + " " + time, "%y%m%d %H%M%S")


def read_paths_from_user_data(
    path: Union[Path, str]
) -> tuple[list[Path], list[Path], Path, Path]:
    """
    Method to read paths to images, baseline images and conig from
    standardized input config file.

    Args:
        path (str or Path): path to user-specific json file.

    Returns:
        list of Path: paths to images
        list of Path: paths to baseline images
        Path: path to config file
        Path: path to results directory
    """
    # Convert to Path
    if isinstance(path, str):
        path = Path(path)

    # Fetch json file
    with open(path, "r") as openfile:
        user_data = json.load(openfile)

    # Define the location for images of C1
    images_folder = Path(user_data["images folder"])
    file_ending = user_data["file ending"]
    all_images = list(sorted(images_folder.glob(file_ending)))

    # Extract basline images and actual images of the injection
    num_baseline_images = user_data["number baseline images"]
    baseline = all_images[:num_baseline_images]
    images = all_images[num_baseline_images:]

    # Path to analysis specific config file
    config = Path(user_data["config"])

    # Path to results directory, create if not existent yet
    results = Path(user_data["results"])
    results.mkdir(parents=True, exist_ok=True)

    return images, baseline, config, results
