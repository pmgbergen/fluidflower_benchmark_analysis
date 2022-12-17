The benchmark initiative collects several experiments for different geometries,
and of different kinds.

In particular in the benchmark geometry (large rig), two sets of experiments
have been performed: Tracer experiments, and CO2 experiments.

In a smaller rig with two different geometries (Albus, Bilbo), CO2 experiments
have been conducted.

All of these experiments contain several runs, which all are analyzed using
DarSIA. The script in this folder together with DarSIA allow to reproduce
the analysis. It includes the preprocessing of images (color, shape, drift,
compaction correction), segmentation of the domain in water, CO2 water, CO2 gas.

Some of the folders contain a further readme with instrutions on how to
run the specific study.

--------------------------------------------------------------------------------

Important user-input is a file which must(!) be called user_data.json. It will
contain all important user/machine-specific information to run the analysis.
In particular, it contains the paths to the folder of all images considered
for the analysis, the file type (note the use of "*." which is necessary.*
Furthermore, the number of baseline image in the folder has to be specified.
Multiple baseline images are beneficial for the construction of tailored
cleaning routines. Next, the location of the config file has to be
specified. Finally, a directory has to be specified in which all results
will be written to file.

A template / an example of the content (feel free to copy and modify) would be:

{
    "images folder": "/home/user/images/c1",
    "file ending": "*.JPG",
    "number baseline images": 10,
    "config": "./config.json"
    "results": "./results"
}

--------------------------------------------------------------------------------

Structure of the analysis folder. The folder contains several subanalysis.

1. Segmentation.

Most of the subsequent analysis is based on a segmentation of the domain into
regions with water, CO2(w), and CO2(g). The segmentation is distributed among
the folders:

  - large_rig: contains the official segmentation routines for C1-C6.
  - albus: contains the official segmentation routines for AC.
  - bilbo: contains the official segmentation routines for BC.

Auxiliary and soon to be deprecated folders:

  - c1-5
  - c1-5-fixed

2. Mass analysis.

Based on the segmentation a 'binary' mass analysis is performed. It is organized
in the folders:

  - mass_analysis: for C
  - mass_analysis_new: for C
  - mass_analysis_bilbo: for BC

3. Segmentation comparison.

To construct 'dragon' images, organized in:

  - segmentation_comparison.

4. Finger analysis.

Anlysis of spatio-temporal evolution of finger tips (number, etc.) is performed
for the large rig in:

  - fingers

5. Compaction/Sand settling.

Volume correlation for different times tracking pre and postcompaction events.

  - compaction_analysis

6. Tracer test.

For the tracer test provided under the benchmark initiative, the continuous
concentration profiles have been extracted. The respective folder:

  - well_test.
