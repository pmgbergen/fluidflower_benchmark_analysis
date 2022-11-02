Each folder represents one run. It contains required scripts to run the analysis.
The main script is analysis.py. To run the script, run:

python analysis.py

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
