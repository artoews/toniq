import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from os import path, makedirs
from pathlib import Path
import sigpy as sp
from time import time

import dicom
from comsi.ge import ScanArchive
from plot import plotVolumes

from util import debug, safe_divide, coord_mats, normalize, equalize


p = argparse.ArgumentParser(description='Field map estimation from MAVRIC-SL data.')
p.add_argument('root', type=str, help='path where outputs are saved')
p.add_argument('-f', '--file', type=str, default=None, help='path to scanarchive file')
p.add_argument('-r', '--reference', type=str, default=None, help='path to corresponding dicom series for shape reference')
# p.add_argument('-e', '--exam_root', type=str, default=None, help='directory where exam data exists in subdirectories')
# p.add_argument('-s', '--series_list', type=str, nargs='+', default=None, help='exam_root subdirectory to use for field map estimation')
p.add_argument('-v', '--verbose', action='count', default=0, help='verbosity level')

def center_of_mass(mass, coords, axis):
    total_mass = np.sum(mass, axis=axis, keepdims=True)
    normalized_mass = np.divide(mass, total_mass, out=np.zeros(mass.shape), where=(total_mass != 0))
    return np.sum(coords * normalized_mass, axis=axis)

def grad_maps(gz, gx, shape, res):
    x, _, z = coord_mats(shape, res=res, loc=(0.5, 0.5, 0.5), offset=0.5)
    gz_map = gz * z
    gx_map = gx * x
    return gx_map, gz_map  # kHz

def rsos(arr, axis):
    return np.sqrt(np.sum(np.abs(arr) ** 2, axis=axis))

def load_dicom_series(path):
    if path is None:
        return None
    files = Path(path).glob('*MRDC*')
    image = dicom.load_series(files)
    return image

def sinc_interpolate(image, shape):
    return np.real(sp.ifft(sp.resize(sp.fft(image), shape)))

def upsample(image, factor, axes):
    for ax in axes:
        image = np.repeat(image, factor, axis=ax)
    return image


if __name__ == '__main__':

    args = p.parse_args()
    
    # set up directory structure
    save_dir = path.join(args.root, 'field')
    if not path.exists(save_dir):
        makedirs(save_dir)

    with open(path.join(save_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    
    if args.reference is not None:
        ref_image = load_dicom_series(args.reference)
        ref_image = np.abs(ref_image.data)

    # load scanarchive data (raw kspace with bins & coils)
    s = ScanArchive(args.file, verbose=True)
    s.save_params(path.join(save_dir, 'seq.yml'))
    t0 = time()
    data, mask, control_table = s.extract_data(flipread=False, verbose=False)
    debug('done extraction', t0)
    mask = np.broadcast_to(mask, (s.nx, s.ny, s.nz, s.nb))
    t0 = time()
    # np.save(path.join(save_dir, 'data.npy'), data)  # (k, b, c)
    # np.save(path.join(save_dir, 'mask.npy'), mask)  # (x, y, z, b)
    
    # zero infill to shape
    shape = (128, 96, 32, 24, 30)
    k = np.zeros(shape, dtype=np.complex128)
    mask = np.repeat(mask[..., None], shape[-1], axis=-1)
    k[mask] = data.ravel()

    # combine coils via complex summation to get bin images
    # ( alternatively, COULD do this: combine coils in image space: compress coil, estimate sensitivity maps, form conjugate coil combination )
    bin_images = sp.ifft(k, axes=(0, 1, 2))
    bin_images = rsos(bin_images, axis=-1)  # combine coils
    bin_images = normalize(bin_images)
    print('bin images', bin_images.shape, bin_images.dtype)

    bin_images_init_size = bin_images.copy()
    new_shape = (128, 128, 32, 24)
    bin_images = np.zeros(new_shape, dtype=bin_images.dtype)
    for i in range(bin_images.shape[-1]):
        im = bin_images_init_size[..., i]
        # not sure about order of these next 3 lines
        im = sp.resize(im, new_shape[:3])
        im = s.gradwarp(im)
        im = np.flip(im, axis=0)
        bin_images[..., i] = im

    # suppress noisy bin voxels
    bin_pct = np.percentile(bin_images, 75, axis=-1, keepdims=True)
    mask = (bin_images > bin_pct)
    bin_images[~mask] = 0

    # field map estimation and save result
    bin_centers = np.linspace(-11.5, 11.5, 24)  # TODO read this from the scan archive
    field = center_of_mass(bin_images, bin_centers, -1)

    # adjust field to remove VAT
    seq_res = np.array([2.4, 2.4, 2.4])  # mm
    seq_shape = new_shape[:3]
    G_cm_to_kHz_mm = 0.42577
    gx = 1.912 * G_cm_to_kHz_mm  # kHz/mm
    gz = 0.795 * G_cm_to_kHz_mm # kHz/mm
    # _, field_vat = grad_maps(gz, gx, seq_shape, seq_res)
    # field = field - field_vat
    # VAT field not needed here if you are going to take the difference from the plastic field anyway

    # print('field at top edge', field[14:18, 64, 15])
    # for i in range(12, 24):
    #     print('bin {} at top edge {}'.format(i, bin_images[14:18, 64, 15, i]))
    # print('bin_centers', bin_centers[12:])

    # form composite image from RSOS bin combination
    image = rsos(bin_images, axis=-1)  # combine bins

    # image = upsample(image, 2, (0, 1))
    # field = upsample(field, 2, (0, 1))

    if args.reference is not None:
        # image = sp.resize(image, (128, 128, 32))
        # field = sp.resize(field, (128, 128, 32))
        # image = sinc_interpolate(image, ref_image.shape)
        # field = sinc_interpolate(field, ref_image.shape)
        ref_image, image = equalize(np.stack((ref_image, image)))
    else:
        image = normalize(image)

    np.save(path.join(save_dir, 'image.npy'), image)
    np.save(path.join(save_dir, 'field.npy'), field)
    debug('done field estimation', t0)

    # plot bin images
    volumes = []
    titles = []
    for i in range(0, shape[3], 4):
        volumes += [bin_images[..., i]]
        titles += ['bin {}'.format(i)]
    fig0, tracker0 = plotVolumes(volumes)
    fig1, tracker1 = plotVolumes(((image - 0.5) * 12, field), vmin=-12, vmax=12)
    fig2, tracker2 = plotVolumes((bin_images[:, :, 15, :],))
    if args.reference is not None:
        fig3, tracker3 = plotVolumes((image, ref_image, np.abs(image - ref_image) + 0.5), titles=('scanarchive', 'dicom', 'difference'))

    plt.show()
    