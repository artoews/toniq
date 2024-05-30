# Image Quality Analysis Software

This page contains all code used in the paper. The [scripts](scripts) folder contains routines to reproduce results from the paper. The [toniq](toniq) folder is a Python package containing general-purpose tooling for image quality analysis studies. This is where the functions for quantitative mapping of non-uniform image quality are implemented, including: intensity artifact, geometric distortion, signal-to-noise ratio, and spatial resolution.

## Installation

We recommend using a package management tool such as conda for working with the TONIQ analysis software. Miniconda3 is recommended and can be downloaded from [here](https://docs.anaconda.com/free/miniconda/).

After installing conda, navigate to the TONIQ analysis folder and run the `setup.sh` script as below. This will create and activate a conda environment `toniq` with all of the required dependencies.

```
cd path/to/toniq/analysis
conda env create -f environment.yml
conda activate toniq
pip install -e .
```

## Getting Started

First check the installation was successful by running the `demo.sh` script as below. This script generates Figure 2 from scratch (running all requisite analyses on the the provided phantom scans). All outputs are saved to a `demo` folder. If the script finishes without error and the new Figure 2 image files match the paper then everything should be working. The script takes a few minutes to run on a modern laptop.

```
cd path/to/toniq/analysis/scripts
bash demo.sh
```

The script `make_figures.sh` runs all requisite analyses to generate the complete set of figures from the paper. It also serves to provide example commands for generating individual figures, e.g. `python figure5.py save_dir`. Note that Figures 5 & 9 are very slow to generate from scratch, and take up the bulk of the long runtime for this script (about 1 hour on a modern laptop). It is recommended to comment out those lines if they are not of interest. To run `make_figures.sh` you must provide a directory to save outputs and a 0/1 indicating whether to run all analyses from scratch (1) or re-use outputs in the provided directory (0).

```
cd path/to/toniq/analysis/scripts
bash make_figures.sh save_dir 1
```

The TONIQ package is documented in-place following the [Google style](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings). See `figure[1-10].py` in [scripts](scripts) for example usage of the TONIQ package.



