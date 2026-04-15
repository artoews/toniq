# Guidance for 3D Printing

## Filament Material

PLA is recommended for several reasons:
- low signal intensity
- low cost
- low toxicity
- negative buoyancy in water

## 3D Printer

Any modern Fused Filament Fabrication (FFF) printer should work.

## Slicing Software

We use [Bambu Studio](https://bambulab.com/en/download/studio) for open-source slicing software but many other options exist.

### Recommended Settings

We recommend the following custom settings based on our experience using Bambu Studio with a [Bambu Lab X1E 3D Printer](https://bambulab.com/en-us/x1e). You may observe small differences in parameter names with other slicing software but the general ideas should still apply.

#### Frame ([frame.STL](frame.STL))

- Infill density: 100%
- Outer brim

#### Lattice Blocks ([lattice-hole-1.STL](lattice-hole-1.STL), [lattice-hole-2.STL](lattice-hole-2.STL), [lattice-hole-array.STL](lattice-hole-array.STL))

- Wall loops: 1
- Top shell layers: 0
- Bottom shell layers: 0
- Sparse infill pattern: gyroid
- Sparse infill density: 10.6% (empirically found to give a 10-mm unit cell)
- Outer brim with 0.2-mm gap (larger gap improves fit in frame)

For better print quality around the rod holes, try reducing the print speed & acceleration. The first layer is particularly sensitive in this regard due to the lack of bottom shell layers. Indeed, for some printers it will be strictly necessary to slow down for a successful print.
