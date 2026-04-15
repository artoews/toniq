# Total Hip Arthroplasty (THA) Design

This design was introduced in the original [TONIQ paper](https://onlinelibrary.wiley.com/doi/10.1002/mrm.30222) to benchmark image quality near a total hip arthroplasty (THA) device. Instructions are provided for [adapting the design](adapting-the-design.md) to a different THA device and/or container.

| Uniform Configuration | Structured Configuration | Structured Configuration<br>(Top Block Removed) |
|---------|---------|---------|
| <img src="https://github.com/user-attachments/assets/0a75f827-5c74-4dc2-b5e2-2eaedd460f2b" width="100%"> | <img src="https://github.com/user-attachments/assets/49dfde4c-1c63-4bbc-9190-939c973c8629" width="100%"> | <img src="https://github.com/user-attachments/assets/0a0e7fc4-b9e4-49dc-ae71-3258ad371b88" width="100%"> |

The above images depict two of the four configurations needed for TONIQ analysis ("Uniform-Metal", "Structured-Metal"). The other two configurations ("Uniform-Plastic", "Structured-Plastic") are realized by replacing the THA device with a 3D-printed plastic replica like the one shown below.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c0aeec65-cfd0-423c-9bd8-7bc7f6e496c8" width="50%" />
</p>

## Tips & Tricks

The large size of this phantom can result in standing EM wave artifacts depending on the signal medium and field strength. We find that water works fine at lower field strengths (e.g. 0.5T), but for 3T we recommend mineral oil.

To minimize bubble formation inside the gyroid lattice, fill the container with your signal medium of choice *before* installing the lattice. This way you can ensure the lattice fills from the bottom up, preventing trapped air.
