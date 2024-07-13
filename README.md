# TONIQ: Toolbox for Non-uniform Image Quality

## Overview

The TONIQ repository provides an ensemble of tools & resources for benchmarking the image quality performance of MRI techniques. TONIQ was initially released in support of [this paper](https://doi.org/10.1002/mrm.30222). If you use any materials from this repository please acknowledge our work with the following citation.

> Toews AR, Lee PK, Nayak KS, Hargreaves BA. Comprehensive assessment of nonuniform image quality: Application to imaging near metal. Magn Reson Med. 2024;1-15. doi: 10.1002/mrm.30222

## Features

The TONIQ repository supports the reproduction and adaptation of work described in the above paper. In particular, the toolbox can allow you to:
- Reproduce all paper figures from the provided image data
- use the TONIQ analysis Python package to generate quantitative maps of image quality factors, including:
  - intensity artifact
  - geometric distortion
  - signal-to-noise ratio
  - spatial resolution
- Build your own modular phantom and use it to benchmark non-uniform image quality

## Orientation

The [scans](scans) folder contains all image data used in the paper, along with practical advice for scanning your own modular phantom.

The [analysis](analysis) folder has everything you need to analyze the image data, including:
- installation guide
- TONIQ Python package
- Python scripts for reproducing each figure in the paper

The [figures](figures) folder contains all figures from the paper.

The [phantom](phantom) folder has everything you need to develop your own modular phantom, including:
- phantom design files & product links
- guidance for adapting the design to another implant
- fabrication instructions (e.g. print settings)

## Correspondence

Please direct any questions or recommendations for the toolbox to Alex Toews (artoews@stanford.edu).

## Disclaimer

The toolbox is intended for research use only and NOT FOR DIAGNOSTIC USE. It comes without any warranty.
