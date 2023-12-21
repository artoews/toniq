import numpy as np
import sigpy as sp

try:
    import matlab.engine as me
    import matlab
except:
    print('Warning: matlab interface could not be imported.')

from matplotlib import pyplot as plt
from scipy.io import loadmat

from ge_scanarchive import MavricSL
from GERecon import Arc, ChannelCombiner, Gradwarp, Orientation, Transformer, ZTransform, Dicom

from plot import plotVolumes
from util import equalize

def extract_data(file, verbose=True):
    # TODO look into this: In Dec 2023 I noticed orchestra example python scripts taking a different 
    # approach that probably dodges the frame skipping issue. See SimpleCartesian3DRecon.py and Arc.py at:
    # /Users/artoews/root/code/tools/orchestra/orchestra-sdk-2.1-1.python/Scripts/
    " Extract raw undersampled k-space into a zero-filled array "
    archive = MavricSL(file)
    shape = (archive.num_readout,
             archive.num_pe,
             archive.num_slices,
             archive.num_bins,
             archive.num_coils)
    control_table = np.zeros((archive.control_count, 3), dtype=int)
    kspace_xyzbc = np.zeros(shape, dtype=np.complex64)
    mask_yzb = np.zeros(shape[1:4], dtype=bool)
    bins_per_pass = int(archive.num_bins / archive.num_passes)
    bin_order = np.argsort(archive.b0_offsets).argsort() # index ordering of bins
    ip = 1
    for ic in range(archive.control_count):  # philip has -1?
        if verbose and np.mod(ic, 1e4) == 0:
            print('Extracting, at control index {} of {}'.format(ic, archive.control_count))
        control = archive.NextControl()
        if control['opcode'] == 1:
            data = np.squeeze(archive.NextFrame())
            while data.shape != (archive.num_readout, archive.num_coils):
                if verbose:
                    print('at control {}: skipping frame for bad shape; expected {} but got {}'.format(ic, (archive.num_readout, archive.num_coils), data.shape))
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

def combine_coils_segfault(archive: MavricSL, kspace_xyzbc):
    " Get bin images from undersampled (zero-filled) coil k-space "
    nx, ny, nz, nb, nc = kspace_xyzbc.shape
    image_xyc = np.zeros((nx, ny, nc))
    image_xyzb = np.zeros((nx, ny, nz, nb))
    arc = Arc(archive.archive.ArcParams())
    channel_combiner = ChannelCombiner(archive.archive.ChannelCombinerParams())
    transformer = Transformer(archive.archive.TransformerParams())
    zTransform = ZTransform(archive.archive.ZTransformParams())
    for ib in range(archive.num_bins):
        k = kspace_xyzbc[:, :, :, ib, :]
        k = arc.Process(k) # TODO check
        k = sp.ifft(k, axes=2, center=False) # raw data format anticipates a non-centered FFT
        # for ic in range(archive.num_coils):
        #     k = zTransform.Execute(k[:, :, :, ic]) # TODO check
        for iz in range(archive.num_slices):
            for ic in range(archive.num_coils):
                k_xy = k[:, :, iz, ic]
                image_xyc[:, :, ic] = sp.ifft(k_xy)
            # image_xyc = transformer.Execute(k[:, :, iz, :]) # TODO check
            image_xyzb[:, :, iz, ib] = np.sqrt(np.sum(image_xyc ** 2))
            # image_xyzb[:, :, iz, ib] = np.abs(channel_combiner.SumOfSquares(image_xyc)) # TODO check
    return image_xyzb

def combine_coils(archive_file, kspace_xyzbc):
    eng = me.start_matlab()
    print('kspace_xyzbc', kspace_xyzbc.shape, kspace_xyzbc.dtype)
    kspace_xyzbc = matlab.double(kspace_xyzbc.astype(np.complex128), is_complex=True)
    # kspace_xyzbc = matlab.single(kspace_xyzbc.astype(np.complex64), is_complex=True)
    print('start engine call')
    image_xyzb = eng.combine_coils(archive_file, kspace_xyzbc)
    print('end engine call')
    print('image_xyzb', image_xyzb.shape, image_xyzb.dtype)
    return image_xyzb

def combine_bins(archive: MavricSL, image_xyzb):
    " Combine bin images into one composite image; includes field-map based 'pixel deblurring'"
    offsets = matlab.int16(archive.b0_offsets())
    image_xyzb = matlab.double(image_xyzb, is_complex=True)
    eng = me.start_matlab()
    # eng.cd(r'myFolder', nargout=0)  # do something like this if combine_bins can't be found
    image_xyz = eng.combine_bins(image_xyzb, offsets)
    print('image_xyz', image_xyz.shape, image_xyz.dtype)
    return image_xyz

def correct_geometry(archive: MavricSL, image_xyz):
    " Apply corrections to address gradient warp and orientation "
    gradwarp = Gradwarp()
    image_xyz = image_xyz.astype('float32')
    corners = (archive.archive.Corners(0), archive.archive.Corners(archive.num_slices-1))
    # fov_scaling = {'FrequencyPixelScaling': 1.0, 'PhasePixelScaling': 1.0}
    fov_scaling = archive.archive.GradwarpParams()
    image_xyz = gradwarp.Execute3D(image_xyz, corners, fov_scaling, gradient="HRMW")
    # TODO see about extracting the coefficients from archive instead of hard-coding the gradient type
    orient = Orientation()
    for iz in range(archive.num_slices):
        orientation = archive.archive.Orientation(iz)
        image_xyz[:, :, iz] = orient.OrientImage(image_xyz[:, :, iz], orientation) 
    return image_xyz

def product_recon(archive: MavricSL):
    kspace_xyzbc, _, _ = extract_data(archive)
    image_xyzb = combine_coils(archive, kspace_xyzbc)
    image_xyz = combine_bins(archive, image_xyzb)
    image_xyz = correct_geometry(archive, image_xyz)
    return image_xyz

if __name__ == '__main__':
    # file = '/Users/artoews/root/code/projects/metal-phantom/orchestra/Data/MAVRICSL/ScanArchive_415723SHMR18_20231021_205205589.h5'
    file = '/Users/artoews/root/data/mri/231021/Series20/ScanArchive_415723SHMR18_20231021_205205589.h5'
    archive = MavricSL(file)
    # image_xyz = product_recon(archive)
    # kspace_xyzbc, _, _ = extract_data(file)
    # np.save('kspace_xyzbc.npy', kspace_xyzbc)
    # kspace_xyzbc = np.load('kspace_xyzbc.npy')
    # print('kspace_xyzbc', kspace_xyzbc.shape, kspace_xyzbc.dtype)
    # image_xyzb = combine_coils(file, kspace_xyzbc)
    
    # validation of correct_geometry
    image = loadmat('/Users/artoews/root/code/projects/metal-phantom/orchestra/MSL/image.mat')['image']
    image_nocorr = loadmat('/Users/artoews/root/code/projects/metal-phantom/orchestra/MSL/image_nocorr.mat')['image_nocorr']
    image_corr = correct_geometry(archive, image_nocorr)
    images = np.stack((image_nocorr, image, image_corr))
    images = equalize(images)
    titles = ('no corr', 'MATLAB corr', 'Python corr', '10x error')
    fig, tracker = plotVolumes((images[0], images[1], images[2], np.abs(images[2]-images[1])), titles=titles)
    plt.show()



    
