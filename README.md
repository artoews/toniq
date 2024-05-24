# Toolbox for Non-uniform Image Quality Analysis

## Overview

The **To**olbox for **N**on-uniform **I**mage **Q**uality Analysis (TONIQ) provides an ensemble of tools & resources for benchmarking Magnetic Resonance Imaging techniques. TONIQ was initially released in support of [this paper](https://github.com/artoews/metal-phantom) (TODO update link when published). If you use any materials from this toolbox please acknowledge our work with the following citation.

> Toews AR, Lee PK, Nayak KS, Hargreaves BA. Comprehensive assessment of non-uniform image quality:
application to imaging near metal. (TODO finish citation when published).

## Features

The toolbox currently supports the reproduction, replication, and adaptation of methods presented in the above paper. In particular, the toolbox can allow you to do any of the following.
- Reproduce all paper figures from the provided image data
- Generate quantitative maps of non-uniform image quality in terms of intensity artifact, geometric distortion, signal-to-noise ratio, and spatial resolution using the TONIQ Python package
- Build your own modular phantom and use it to benchmark a Magnetic Resonance Imaging technique of your choice

## Organization

Here's where to find everything you might be looking for.

phantom/
- phantom design files & product links
- fabrication instructions
- guidance for adapting the modular phantom design

scans/
- all image data used in the paper
- practical advice for scanning the modular phantom

analysis/
- TONIQ Python package
- installation guide
- Python scripts for reproducing each figure in the paper

figures/
- all paper figures

## Correspondence

Please direct any questions or recommendations for the toolbox to Alex Toews (artoews@stanford.edu).

## Disclaimer

The toolbox is intended for research use only and NOT FOR DIAGNOSTIC USE. It comes without any warranty.
