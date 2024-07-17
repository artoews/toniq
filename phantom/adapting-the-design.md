# Adapting the Design

The following instructions describe how to modify the phantom design to fit a different container or implant. Note a SolidWorks License is required to carry out these changes.

## A. Prepare models of new components

1. Measure the interior dimensions of the new container (length, width).
2. Acquire surface model of the implant in STL format.
    * If structured light scanning is used (e.g. [DAVID SLS-2](https://productrealization.stanford.edu/processes/3d-scanning)), then spray chalk is recommended for a temporary matte surface finish.
3. Refine the STL surface model.
    * Install free mesh processing software [Autodesk Meshmixer](https://meshmixer.com/download.html).
    * Open STL in Meshmixer.
    * Check the model scaling is correct via "Analysis→Units/Dimensions". Update units as necessary (choosing option to preserve XYZ coordinates).
    * Align model with cardinal planes via “Edit→Align” and “Edit→Transform”, as needed.
    * Repair model by erasing nuisance features (e.g. holes) via “Select→Delete” and “Analysis→Inspector→Auto Repair All”.
4. Generate a toleranced copy of the surface model.
    * In Meshmixer, use “Select→Edit→Offset” with a reasonably small offset (e.g. 0.5 mm). This will create a second, slightly larger solid body in the same file as the original one. These two overlapping solid bodies will be separated in a later step. The original solid body ("implant surface") is for printing the plastic replica implant. The larger solid body ("implant surface offset") is for making the implant cavity feature in the block assembly.
5. Compress surface models.
    * In Meshmixer, “Select→Edit→Reduce” with 0.25 mm Max Deviation (shape preserving).
    * Save as a new file in STL format.
6. Separate the two surface models.
    * Open the new STL file in SolidWorks. In the Open File window, click Options to ensure the STL will be imported as a Solid Body with the correct units. The Options button will appear when you select an STL file.
    * Create separate part files for each one of the two surfaces by deleting the other solid body and saving as a new *.SLDPRT file.

## B. Update design files

In this section *italics* are used to indicate the names of SolidWorks Global Variables to be updated.

1. Update main assembly (phantom.sldasm)
    * Replace implant (surface and offset parts) with new models. “File → Replace” for each of implant-surface.sldprt and implant-offset.sldprt.
    * Keep phantom.sldasm open for subsequent changes to ensure synchronization of inter-file dependencies.
2. Update block (block.sldprt).
    * Set gyroid cell size, as needed (*cell-size*).
    * Set block dimensions in units of cells, as needed (*nx*, *ny*, *nz*).
3. Update frame (frame.sldprt).
    * Match lengths of fixation arms to new container dimensions (*box_length*, *box_width*).
    * Adjust relative position of block on frame, as needed (*offset_SI*).

## C. Generate files for printing

1. For each printed component (block.sldprt, frame.sldprt, implant.sldprt and optionally line-pairs-xy.sldasm):
    * Open file in SolidWorks.
    * Export as STL: “File → Save As → Save as type: STL”.
2. Part file block.sldprt can be toggled between top/bottom block and structured/uniform interior (defaults to bottom & structured). Two additional blocks must be generated to build the uniform and structured phantom configurations.
    * Bottom block for uniform signal configurations.
        * In block.sldprt, unsuppress Feature folder “FOR-UNIFORM-CONFIG”.
        * Export STL as in step 1.
        * Revert feature suppression to the initial state.
    * Top block for structured signal configurations.
        * In phantom.sldasm, suppress Mate “FOR-BOTTOM-BLOCK”.
        * In phantom.sldasm, unsuppress Mate “FOR-TOP-BLOCK”.
        * In block.sldprt, export STL as in step 1.
        * Revert mates to their initial state.
