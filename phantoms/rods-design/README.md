# Rods Design

This design was introduced after the original [TONIQ paper](https://onlinelibrary.wiley.com/doi/10.1002/mrm.30222) to improve usability. The metal rods are intended to imitate various orthopedic implants like intramedullary nails, pedicle screws, and spinal fixation rods.

<p align="center">
  <img src="https://github.com/user-attachments/assets/07d056fb-9a8c-4d7a-b57e-21729ca6ed77" width="48%" />
  <img src="https://github.com/user-attachments/assets/79bd1653-b9c2-4c9c-a27c-d1e01999aa59" width="48%" />
</p>

The multi-rod configuration shown above is intended for qualitative analysis only. To perform a quantitative (TONIQ) analysis, a single-rod configuration should be used, following the paradigm of the original TONIQ paper. STL files are provided for both multi-hole and single-hole options.

## Usability

This design improves upon the usability of the THA design in several ways:
- Print & play: No 3D surface scanning or CAD tools are needed to replicate the design. Rod stock in standard dimensions is widely available.
- Water works: The smaller form factor allows for plain tap water to be used as the signal medium without significant interference from standing EM waves.
- 2-for-1: The "Structured" and "Uniform" configurations can be realized simultaneously (vertically stacked), owing to the symmetry of the metal rods. This substantially reduces the amount of scanning required for TONIQ analysis (three scans instead of five).

## Materials

We use 6" long metal rods in two diameters: 1/4" (~6mm) and 1/8" (~3mm). Commercial product links are collected [here](product-links.md).

Three materials are recommended to cover the range of magnetic susceptibility values found in orthopedic implants.
- Ceramic Alumina (-10 ppm)
- Grade 5 Titanium (+200 ppm)
- 316L Stainless Steel (+4000 ppm)

*All magnetic susceptibility values are approximate - actual values will depend on the exact alloy blend and preparation.*

## Tips & Tricks

O-rings can be used to hold the rods above the container floor, as shown below. This is helpful for analyzing image quality at the end of the rods, where artifacts tend to be the worst.
<p align="center">
  <img src="https://github.com/user-attachments/assets/79767a6f-9af5-4235-989f-5d1082d1a147" width="50%" />
</p>

To minimize bubble formation inside the gyroid lattice, fill the container with water *before* installing the lattice. This way you can ensure the lattice fills from the bottom, preventing trapped air.
