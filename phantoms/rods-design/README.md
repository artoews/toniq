# Rods Design

This design was introduced after the original TONIQ paper to improve usability. The metal rods are intended to imitate various orthopedic implants including pedicle screws, spinal fixation rods, and intramedullary nails.

<p align="center">
  <img src="https://github.com/user-attachments/assets/07d056fb-9a8c-4d7a-b57e-21729ca6ed77" width="48%" />
  <img src="https://github.com/user-attachments/assets/79bd1653-b9c2-4c9c-a27c-d1e01999aa59" width="48%" />
</p>

The multi-rod configuration shown above is intended for qualitative analysis only. To perform a quantitative (TONIQ) analysis, a single-rod configuration should be used, following the paradigm of the original TONIQ paper. STL files are provided for both multi-hole and single-hole options.

## Usability

This design improves upon the usability of the THA design in several ways:
- Print & play: No 3D surface scanning or CAD tools are needed to replicate the design, since rod stock is widely available (unlike orthopedic implants)
- Water works: The small form factor allows for plain tap water to be used as the signal medium without significant interference from standing waves. No more messy mineral oil.
- 2-for-1: The "Structured" and "Uniform" configurations can be realized simultaneously (vertically stacked), owing to the symmetry of the metal rods. This reduces the number of scans required for TONIQ analysis from five to three.

## Materials

We use 6" long metal rods in two diameters, 1/4" (~6mm) and 1/8" (~3mm).

Three materials are recommended to span the range of magnetic susceptibility values found in orthopedic implants.
- Ceramic Alumina (-10 ppm)
- Grade 5 Titanium (+200 ppm)
- 316L Stainless Steel (+4000 ppm)

*All magnetic susceptibility values are approximate - actual values will depend on the exact alloy blend and preparation.*

## Tips & Tricks

O-rings can be used to hold the rods above the container floor, as shown below. This is helpful for interrogating the difference in artifact presentation between the middle and ends of a rod.
<p align="center">
  <img src="https://github.com/user-attachments/assets/79767a6f-9af5-4235-989f-5d1082d1a147" width="50%" />
</p>

To minimize bubble formation inside the gyroid lattice, fill the container with water *before* installing the lattice. This way you can ensure the lattice fills from the bottom, preventing trapped air.
