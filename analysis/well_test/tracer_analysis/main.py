"""
The well test data is stored in analysis/well_test. However, a slightly
wrong injection start time has been chosen resulting in wrong relative.
This is corrected here.

In addition the files are stored as csv (without header).
"""
from pathlib import Path
import numpy as np

# The output of the well test is structured in 3 parts.
# 1. First two phases with 6 sub phases each, alternating between
# injection and pause.
# 2. Break overnight.
# 3. Third phase similar to the first two.

pt1 = Path("../../well_test/results_pt1").glob("*npz")
pt2 = Path("../../well_test/results_pt2").glob("*npz")
pt3 = Path("../../well_test/results_pt3").glob("*npz")

# The file data_DSC00109.npz is identified with time = 0.
# Fetch the correpsonding time which is stored in the second
# component, while the image is stored in the first component.
ref_file = Path("../../well_test/results_pt1/data_DSC00109.npz")
ref_data = np.load(ref_file)
ref_time = ref_data["arr_1"]

# Iterate over all files and update the time, store the files
# again as npz files. In addition, store csv files.
def convert_data(folder: Path) -> None:

    for path in folder:
    
        # Fetch data and time
        data = np.load(path)
        array = data["arr_0"]
        time = data["arr_1"]
    
        # Update time
        new_time = time - ref_time
    
        # Fetch path without ending
        data_name = Path(path.name).with_suffix("")

        # Store again
        np.savez(Path(f"./npz_data/{data_name}.npz"), array, new_time)
    
        # Store as csv with time in the header
        header = f"Relative time in seconds {str(new_time)}."
        np.savetxt(Path(f"./csv_data/{data_name}.csv"), array, fmt="%.3f", delimiter=",", header=header)

# Run conversion on all folders
convert_data(pt1)
convert_data(pt2)
convert_data(pt3)
