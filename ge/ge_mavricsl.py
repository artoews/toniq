import matlab
import matlab.engine as me
import matplotlib.pyplot as plt
import numpy as np

from GERecon import Gradwarp, Orientation

from ge_scanarchive import MavricSL
from plot import plotVolumes

# NOTE: the size of data arrays passed between Python and MATLAB is limited to 2 GB
# https://www.mathworks.com/help/matlab/matlab_external/limitations-to-the-matlab-engine-for-python.html

def get_bins_from_archive(archive_file, matlab_root='matlab'):
    " Reconstruct bin images from raw k-space data in scan archive "
    # TODO the extraction could be done in Python, and cleaner than in ge_mavricsl_old.py:
    # - In Dec 2023 I noticed orchestra example python scripts taking a different approach that probably dodges the frame skipping issue
    # - See SimpleCartesian3DRecon.py and Arc.py at /Users/artoews/root/code/tools/orchestra/orchestra-sdk-2.1-1.python/Scripts/
    eng = me.start_matlab()
    eng.cd(matlab_root, nargout=0)
    image_xyzb, offsets = eng.get_bins_from_archive(archive_file, nargout=2)
    image_xyzb = np.asarray(image_xyzb).copy()
    offsets = np.asarray(offsets).copy()
    return image_xyzb, offsets

def combine_bins(image_xyzb, offsets, matlab_root='matlab'):
    " Combine bin images into one composite image via RSOS; includes field-map based 'pixel deblurring' "
    eng = me.start_matlab()
    eng.cd(matlab_root, nargout=0)
    image_xyzb = matlab.double(np.abs(image_xyzb))
    offsets = matlab.double(offsets)
    image_xyz = eng.combine_bins(image_xyzb, offsets)
    image_xyz = np.asarray(image_xyz).copy()
    return image_xyz

def correct_geometry(archive_file, image_xyz, matlab_root='matlab'):
    " Apply geometric transformations to correct gradient warp and orientation; uses MATLAB Orchestra"
    eng = me.start_matlab()
    eng.cd(matlab_root, nargout=0)
    image_xyz = matlab.double(image_xyz)
    image_xyz = eng.correct_geometry(archive_file, image_xyz)
    image_xyz = np.asarray(image_xyz).copy()
    return image_xyz

def correct_geometry_in_python(archive: MavricSL, image_xyz):
    " Apply geometric transformations to correct gradient warp and orientation; uses Python Orchestra"
    " (should be equivalent to MATLAB implemenation above) "
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

def product_recon(archive_file, matlab_root='matlab'):
    image_xyzb, offsets = get_bins_from_archive(archive_file, matlab_root=matlab_root)
    image_xyz = combine_bins(image_xyzb, offsets, matlab_root=matlab_root)
    image_xyz = correct_geometry(archive_file, image_xyz, matlab_root=matlab_root)
    return image_xyz

if __name__ == '__main__':

    archive_file = '/Users/artoews/root/data/mri/231021/Series20/ScanArchive_415723SHMR18_20231021_205205589.h5'
    load_from_file = False

    if load_from_file:
        image_xyzb = np.load('test-image_xyzb.npy')
        image_xyz = np.load('test-image_xyz_uncorrected.npy')
    else:
        print('Getting bins from archive')
        image_xyzb, offsets = get_bins_from_archive(archive_file)
        np.save('test-image_xyzb', image_xyzb)
        np.save('test-offsets', offsets)
        print('image_xyzb', image_xyzb.shape, image_xyzb.dtype)

        print('Combining bins')
        image_xyz = combine_bins(image_xyzb, offsets)
        np.save('test-image_xyz_uncorrected', image_xyz)
        print('image_xyz', image_xyz.shape, image_xyz.dtype)

    print('Correcting geometry')
    image_xyz_uncorrected = image_xyz
    image_xyz = correct_geometry(archive_file, image_xyz_uncorrected)
    np.save('test-image_xyz', image_xyz)
    print('image_xyz', image_xyz.shape, image_xyz.dtype)

    # plotting
    image_xyzb = np.abs(image_xyzb)
    image_xyzb = image_xyzb / np.max(image_xyzb)
    image_xyz_uncorrected = image_xyz_uncorrected / np.max(image_xyz)
    image_xyz = image_xyz / np.max(image_xyz)
    fig0, tracker0 = plotVolumes(list(np.moveaxis(image_xyzb, -1, 0)), nrows=4, ncols=image_xyzb.shape[-1]//4) # all bin images, with geometric correction
    fig1, tracker1 = plotVolumes((image_xyz_uncorrected, image_xyz), ('Before correction', 'After correction')) # composite image before and after geometric correction
    plt.show()
