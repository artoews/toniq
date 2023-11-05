import json
import matplotlib.pyplot as plt
import numpy as np
from os import path, makedirs
from pathlib import Path
import scipy.ndimage as ndi
from skimage import morphology
import seaborn as sns
import sigpy as sp
from time import time
import analysis
import dicom
from plot import plotVolumes, overlay_mask
import register
import distortion

from util import safe_divide, masked_copy

def load_outputs(root, subfolder):
    data = np.load(path.join(root, subfolder, 'outputs.npz'))
    for var in data:
        globals()[var] = data[var]

# scan times
# FSE @ 500 kHz = 1:21 (81s)
# FSE @ 250 kHz = 1:21 (81s)
# FSE @ 125 kHz = 1:21 (81s)
# FSE @ 62  KHz = 1:53 (113s)
# MSL @ 250 kHz = 4:29 (269s)
# MSL @ 125 kHz = 6:44 (404s)

# identify the data folders
fse_dir = '/Users/artoews/root/code/projects/metal-phantom/demo-msl-250'
msl_dir = '/Users/artoews/root/code/projects/metal-phantom/demo-msl-125'
seqs = ['250 kHz', '125 kHz']
short_seqs = ['250', '125']
scan_times = [404, 269]

# fse_dir = '/Users/artoews/root/code/projects/metal-phantom/demo-fse-125'
# msl_dir = '/Users/artoews/root/code/projects/metal-phantom/demo-msl-125'
# fse_dir = '/Users/artoews/root/code/projects/metal-phantom/tmp'
# msl_dir = '/Users/artoews/root/code/projects/metal-phantom/msl-demo'
# seqs = ['2D FSE', 'MAVRIC-SL']
# short_seqs = ['FSE', 'MSL']
# scan_times = [81, 269]
dirs = [fse_dir, msl_dir]

# setup the figure
fig, axes = plt.subplots(   nrows=len(dirs)+1,
                            ncols=10,
                            figsize=(22, 10),
                            gridspec_kw={'width_ratios': [1, 1, 1, 0.1, 1, 0.1, 1, 0.1, 1, 0.1]})

for i in range(axes.shape[0]):
    for j in range(axes.shape[1]):
        if i == axes.shape[0] - 1 and j in [2, 4, 6, 8]:
            continue
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
plt.delaxes(axes[-1, 0])
plt.delaxes(axes[-1, 1])
plt.delaxes(axes[-1, 3])
plt.delaxes(axes[-1, 5])
plt.delaxes(axes[-1, 7])
plt.delaxes(axes[-1, 9])
# axes[-1, 2].axvspan(-0.3, 0.3, facecolor='0.8', alpha=0.5, zorder=-100)
axes[-1, 2].set_yticks([])
axes[-1, 4].set_yticks([])
axes[-1, 6].set_yticks([])
axes[-1, 8].set_yticks([])


# plot settings
slc = (slice(None), slice(None), 15)
fs = 20

artifacts_all = []

for i in range(len(dirs)):

    load_outputs(dirs[i], 'artifact')
    print('loaded artifact outputs for {} @ RBW={:.3g}kHz'.format(seqs[i], rbw[0]))

    if i == 0:
        axes[i, 0].set_title('Plastic', fontsize=fs)
        axes[i, 1].set_title('Metal', fontsize=fs)
        axes[i, 2].set_title('Artifact', fontsize=fs)
        axes[i, 4].set_title('Distortion', fontsize=fs)
        axes[i, 6].set_title('Resolution', fontsize=fs)
        axes[i, 8].set_title('SNR', fontsize=fs)

    plastic_image = images[0][slc]
    print('got plastic image @ RBW={:.3g}kHz'.format(rbw[0]))
    axes[i, 0].imshow(plastic_image, vmin=0, vmax=1, cmap='gray')

    axes[i, 0].set_ylabel(seqs[i], fontsize=fs)
    
    metal_image = images[1][slc]
    axes[i, 1].imshow(metal_image, vmin=0, vmax=1, cmap='gray')

    intensity_map = maps_artifact[0][slc]
    im = axes[i, 2].imshow(intensity_map, vmin=-1, vmax=1, cmap='RdBu_r')
    plt.colorbar(im, cax=axes[i, 3], ticks=[-1, 0, 1], label='Relative Error')
    axes[i, 3].yaxis.set_label_position('left')
    artifacts_all += [maps_artifact[0].ravel()]
    if i == 0:
        mask_artifact = (np.abs(artifacts_all[0]) > 0.3)
    else:
        mask_artifact = np.logical_or(mask_artifact, np.abs(artifacts_all[0]) > 0.3)

    load_outputs(dirs[i], 'distortion')
    print('loaded distortion outputs for {} @ RBW={:.3g}kHz'.format(seqs[i], rbw[1]))
    result_mask = (results_masked[0][slc] != 0)
    measured_deformation = -deformation_fields[0][..., 0][slc]
    measured_deformation = measured_deformation * result_mask
    im = axes[i, 4].imshow(measured_deformation, vmin=-4, vmax=4, cmap='RdBu_r')
    plt.colorbar(im, cax=axes[i, 5], ticks=[-4, -2, 0, 2, 4], label='Readout Displacement (pixels)')
    axes[i, 5].yaxis.set_label_position('left')
    axes[-1, 4].hist(measured_deformation[np.abs(measured_deformation)>0].ravel(), bins=np.linspace(-2, 2, 21), alpha=0.5, label=seqs[i])

    load_outputs(dirs[i], 'resolution')
    print('loaded resolution outputs')
    res_x = fwhms[0][..., 0][slc]
    im = axes[i, 6].imshow(res_x, vmin=0, vmax=2, cmap='viridis')
    plt.colorbar(im, cax=axes[i, 7], ticks=[0, 1, 2], label='Readout FWHM (pixels)')
    axes[i, 7].yaxis.set_label_position('left')
    axes[-1, 6].hist(res_x[res_x>0].ravel(), bins=np.linspace(0, 2, 21), alpha=0.5, label=seqs[i])

    load_outputs(dirs[i], 'noise')
    print('loaded noise outputs for {} @ RBW={:.3g}kHz'.format(seqs[i], rbw[0]))
    snr = snrs[0][slc]
    im = axes[i, 8].imshow(snr, vmin=0, vmax=150, cmap='viridis')
    plt.colorbar(im, cax=axes[i, 9], ticks=[0, 50, 100, 150])
    axes[i, 9].yaxis.set_label_position('left')
    axes[-1, 8].hist(snr[snr>0].ravel() / scan_times[i], bins=np.linspace(0, 1, 21), alpha=0.5, label=seqs[i])

for i in range(len(dirs)):
    axes[-1, 2].hist(artifacts_all[i][mask_artifact].ravel(), bins=np.linspace(-1, 1, 21), alpha=0.5, label=seqs[i])
axes[-1, 2].legend()
axes[-1, 2].set_ylabel('Voxels')
axes[-1, 2].set_xlabel('Relative Error')
axes[-1, 2].set_title('Inside Artifact Region')

# axes[-1, 4].legend()
axes[-1, 4].set_ylabel('Voxels')
axes[-1, 4].set_xlabel('Readout Displacement (pixels)')
axes[-1, 4].set_title('Outside Artifact Region')

# axes[-1, 6].legend()
axes[-1, 6].set_ylabel('Voxels')
axes[-1, 6].set_xlabel('Readout FWHM (pixels)')
axes[-1, 6].set_title('Entire Signal Region')

# axes[-1, 8].legend()
axes[-1, 8].set_ylabel('Voxels')
axes[-1, 8].set_xlabel('SNR / seconds of scan time')
axes[-1, 8].set_title('Entire Signal Region')

plt.savefig(path.join(dirs[i], 'demo_comparison_{}_{}.png'.format(*short_seqs)))


for i in range(len(dirs)):
    load_outputs(dirs[i], 'artifact')
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3), gridspec_kw={'width_ratios': [1, 1, 1, 0.1]})

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plastic_image = images[0][slc]
    axes[0].imshow(plastic_image, vmin=0, vmax=1, cmap='gray')
    metal_image = images[1][slc]
    axes[1].imshow(metal_image, vmin=0, vmax=1, cmap='gray')

    intensity_map = maps_artifact[0][slc]
    im = axes[2].imshow(intensity_map, vmin=-1, vmax=1, cmap='RdBu_r')
    plt.colorbar(im, cax=axes[3], ticks=[-1, 0, 1], label='Relative Error')
    axes[3].yaxis.set_label_position('left')

    axes[0].set_title('Plastic', fontsize=fs)
    axes[1].set_title('Metal', fontsize=fs)
    axes[2].set_title('Artifact Map', fontsize=fs)

    plt.savefig(path.join(dirs[i], 'preview_{}.png'.format(short_seqs[i])))

plt.show()
