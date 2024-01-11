import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from os import path, makedirs
import scipy.ndimage as ndi
from skimage import morphology

from field import estimate_field
from ge_mavricsl import get_bins_from_archive, combine_bins, correct_geometry
from plot import plotVolumes
from util import equalize

# TODO VAT field compensation, though not necessary for a differential measurement of field, should be added eventually for completeness
# for this you will need to know the gz gradient, e.g. 
# G_cm_to_kHz_mm = 0.4257
# gx = 1.912 * G_cm_to_kHz_mm  # kHz/mm
# gz = 0.795 * G_cm_to_kHz_mm # kHz/mm

p = argparse.ArgumentParser(description='Differential field map estimation from MAVRIC-SL data.')
p.add_argument('root', type=str, help='path where outputs are saved')
p.add_argument('-p', '--plastic', type=str, default=None, help='path to scanarchive file for plastic implant')
p.add_argument('-m', '--metal', type=str, default=None, help='path to scanarchive file for metal implant')

def clean(image_xyzb, pct=75):
    " Zero out empty bin voxels; serves to reduce bias on center-of-mass field measurement "
    thresh = np.percentile(image_xyzb, pct, axis=-1, keepdims=True)
    image_xyzb[image_xyzb < thresh] = 0
    return image_xyzb

def get_image_and_field(file):
    image_xyzb, offsets = get_bins_from_archive(file)
    image_xyz = combine_bins(image_xyzb, offsets)
    image_xyz = correct_geometry(file, image_xyz)
    image_xyzb = clean(image_xyzb)
    field_xyz = estimate_field(image_xyzb, offsets)
    field_xyz = correct_geometry(file, field_xyz)
    return image_xyz, field_xyz

if __name__ == '__main__':

    args = p.parse_args()
    save_dir = path.join(args.root, 'field')
    if not path.exists(save_dir):
        makedirs(save_dir)
    with open(path.join(save_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    
    if args.plastic is not None and args.metal is not None:
        image_plastic, field_plastic = get_image_and_field(args.plastic)
        image_metal, field_metal = get_image_and_field(args.metal)
        image_plastic, image_metal = equalize((image_plastic, image_metal))
        np.savez(path.join(save_dir, 'outputs.npz'),
                         image_plastic=image_plastic,
                         field_plastic=field_plastic,
                         image_metal=image_metal,
                         field_metal=field_metal,
                         )
    else:
        data = np.load(path.join(save_dir, 'outputs.npz'))
        for var in data:
            globals()[var] = data[var]

    field_diff = field_metal - field_plastic
    field_diff_filtered = ndi.median_filter(field_diff, footprint=morphology.ball(2))
    np.save(path.join(save_dir, 'field_diff_Hz.npy'), field_diff_filtered)

    # plotting
    vols = (image_plastic, image_metal)
    titles=('plastic', 'metal')
    fig0, tracker0 = plotVolumes(vols, titles=titles)
    vols = (field_plastic, field_metal, field_diff, field_diff_filtered)
    titles=('plastic', 'metal', 'difference', 'difference, filtered')
    fig1, tracker1 = plotVolumes(vols, titles=titles, vmin=-12e3, vmax=12e3, cmap='RdBu_r', cbar=True)
    plt.show()
