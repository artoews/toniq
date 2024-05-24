## Adapting the Design

The following instructions describe how to adapt the phantom design files for a different container and/or implant. A SolidWorks License is required to carry out these changes. Italics are used to indicate the names of SolidWorks Global Variables.

### A. Measure new components

1. Measure the interior dimensions of the new container (length, width).
2. Acquire surface model of the implant in STL format.
    * If structured light scanning is used (e.g. [DAVID SLS-2](https://productrealization.stanford.edu/processes/3d-scanning)), then spray chalk is recommended for a temporary matte surface finish.
3. Refine the STL surface model.
    * Install free mesh processing software [Autodesk Meshmixer](https://meshmixer.com/download.html).
    * Open STL in Meshmixer.
    * Align model with cardinal planes via “Edit→Align” and “Edit→Transform”, as needed.
    * Repair model by erasing nuisance features (e.g. holes) via “Select→Delete” and “Analysis→Inspector→Auto Repair All”.
5. Generate a toleranced copy of the surface model.
    * In Meshmixer, use “Select→Edit→Offset” with a reasonably small offset (e.g. 0.5 mm). This will result in two solid bodies, to be separated in a later step.
        * Implant surface, for printing plastic replica.
        * Implant surface offset, for making the implant cavity feature in the block assembly.
6. Compress surface models.
    * In Meshmixer, “Select→Edit→Reduce” with 0.25 mm Max Deviation (shape preserving).
    * Save as a new file in STL format.
7. Separate the two surface models.
    * Open the new STL file in SolidWorks.
    * Create separate part files for each one of the two surfaces by deleting the other solid body and saving as a new *.SLDPRT file.

### B. Update design files

1. Update main assembly (phantom.sldasm)
    * Replace implant (surface and offset parts) with new models. “File → Replace” for each of implant-surface.sldprt and implant-offset.sldprt.
    * Keep phantom.sldasm open for subsequent changes to ensure synchronization of inter-file dependencies.
2. Update block (block.sldprt).
    * Set gyroid cell size, as needed (*cell-size*).
    * Set block dimensions in units of cells, as needed (*nx*, *ny*, *nz*).
3. Update frame (frame.sldprt).
    * Match lengths of fixation arms to new container dimensions (*box_length*, *box_height*).
    * Adjust relative position of block on frame, as needed (*offset_SI*).

### C. Generate models for printing

1. For each printed component (block.sldprt, frame.sldprt, implant.sldprt and optionally line-pairs-xy.sldasm):
    * Open file in SolidWorks.
    * Export as STL: “File → Save As → Save as type: STL”.
2. Generate two additional models from block.sldprt:
    * Bottom block for uniform signal configurations. In block.sldprt:
        * Unsuppress Feature folder “FOR-UNIFORM-CONFIG”.
        * Export STL as above.
        * Revert feature suppression to restore initial state.
    * Top block for structured signal configurations. In phantom.sldasm:
        * Suppress Mate “FOR-BOTTOM-BLOCK”.
        * Unsuppress Mate “FOR-TOP-BLOCK”.
        * In block.sldprt, export STL as above.
        * Revert mate suppression to restore initial state.
