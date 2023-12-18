import numpy as np
import sigpy as sp

from GERecon import Arc, ChannelCombiner, Gradwarp, Orientation, Transformer, Dicom

from ge_scanarchive import MavricSL

def extract_data(archive: MavricSL, verbose=True):
    " Get k-space from MAVRIC-SL scan archive as a zero-filled numpy array "
    control_table = np.zeros((archive.control_count, 3), dtype=int)
    kspace_xyzbc = np.zeros(archive.shape, dtype=np.complex64)
    mask_yzb = np.zeros(archive.shape[1:4], dtype=bool)
    bins_per_pass = int(archive.num_bins / archive.num_passes)
    bin_order = np.argsort(archive.b0_offsets[:archive.num_bins]).argsort() # index ordering of bins
    ip = 1
    for ic in range(archive.control_count):  # philip has -1?
        if verbose and np.mod(ic, 1e4) == 0:
            print('Extracting, at control index {} of {}'.format(ic, archive.control_count))
        control = archive.NextControl()
        if control['opcode'] == 1:
            data = np.squeeze(archive.NextFrame())
            while data.shape != (archive.num_readout, archive.num_coils):
                if verbose:
                    print('Skipping frame {} for bad shape; expected {} but got {}'.format((archive.num_readout, archive.num_coils), data.shape))
                data = np.squeeze(archive.NextFrame())
                # TODO should Control be advanced as well, via NextControl()? And increment ic? if so then replace "NextFrame()" with "continue"?
            iy = control['viewNum'] - 1
            iz = control['sliceNum']
            # iz_alternate = self.archive.Info(control['sliceNum'], ip)['sliceNumber']
            ib = bin_order[control['echoNum'] + (ip - 1) * bins_per_pass]
            kspace_xyzbc[:, iy, iz, ib, :] = data
            mask_yzb[iy, iz, ib] = True
            control_table[ic, :] = [ib, iz, iy]
        elif control['opcode'] == 0:
            if ip < archive.num_passes:
                ip += 1
    return kspace_xyzbc, mask_yzb, control_table

def combine_coils(archive: MavricSL, kspace_xyzbc):
    " Get bin images from zero-filled coil k-space "
    nx, ny, nz, nb, nc = kspace_xyzbc.shape
    image_xyc = np.zeros((nx, ny, nc))
    image_xyzb = np.zeros((nx, ny, nz, nb))
    for ib in range(archive.num_bins):
        k = kspace_xyzbc[:, :, :, ib, :]
        # TODO apply Arc.Synthesize to k
        k = sp.ifft(k, axes=2, center=False) # raw data format anticipates a non-centered FFT
        for iz in range(archive.num_slices):
            for ic in range(archive.num_coils):
                k_xy = k[:, :, iz, ic]
                image_xyc[:, :, ic] = sp.ifft(k_xy)
                # TODO apply GE Transform to each k[:, :, iz, ic]
            image_xyzb[:, :, iz, ib] = np.sqrt(np.sum(image_xyc ** 2))
            # TODO apply GE SumOfSquares to each image_xyc
    return image_xyzb

def combine_bins():
    # TODO
    pass

def correct_geometry():
    # TODO
    pass

def product_recon(archive: MavricSL):
    kspace_xyzbc, _, _ = extract_data(archive)
    image_xyzb = combine_coils(archive, kspace_xyzbc)
    # TODO implement combine_bins (bin correction & combination)
    # TODO implement correct_geometry (gradwarp & orient)
