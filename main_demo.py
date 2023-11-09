import json
import matplotlib.pyplot as plt
import numpy as np
from os import path, makedirs
from pathlib import Path
import scipy.ndimage as ndi
import scipy.stats as stats
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
fse_dir = '/Users/artoews/root/code/projects/metal-phantom/demo-fse-250'
msl_dir = ['/Users/artoews/root/code/projects/metal-phantom/demo-msl-250', '/Users/artoews/root/code/projects/metal-phantom/demo-msl-125']
seqs = ['FSE 250kHz', 'MSL 250kHz', 'MSL 125kHz']
seqs2 = ['FSE\nRBW=250kHz', 'MAVRIC-SL\nRBW=250kHz', 'MAVRIC-SL\nRBW=125kHz']
short_seqs = ['F250', 'M250', 'M125']
scan_times = [81, 269, 404]
# colors = ['black', 'red', 'blue']
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
styles = ['dotted', 'solid', 'dashed']

# fse_dir = '/Users/artoews/root/code/projects/metal-phantom/demo-fse-125'
# msl_dir = '/Users/artoews/root/code/projects/metal-phantom/demo-msl-125'
# fse_dir = '/Users/artoews/root/code/projects/metal-phantom/tmp'
# msl_dir = '/Users/artoews/root/code/projects/metal-phantom/msl-demo'
# seqs = ['2D FSE', 'MAVRIC-SL']
# short_seqs = ['FSE', 'MSL']
# scan_times = [81, 269]
dirs = [fse_dir] + msl_dir

# setup the figure
fig, axes = plt.subplots(   nrows=len(dirs)+1,
                            ncols=10,
                            figsize=(24, 12),
                            gridspec_kw={'width_ratios': [1, 1, 1, 0.1, 1, 0.1, 1, 0.1, 1, 0.1]})

for i in range(axes.shape[0]):
    for j in range(axes.shape[1]):
        if i == axes.shape[0] - 1 and j in [2, 4, 6, 8]:
            continue
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
for i in range(1, axes.shape[0]):
    plt.delaxes(axes[i, 3])
    plt.delaxes(axes[i, 5])
    plt.delaxes(axes[i, 7])
    plt.delaxes(axes[i, 9])
plt.delaxes(axes[-1, 0])
plt.delaxes(axes[-1, 1])
# plt.delaxes(axes[-1, 3])
# plt.delaxes(axes[-1, 5])
# plt.delaxes(axes[-1, 7])
# plt.delaxes(axes[-1, 9])
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
        axes[i, 2].set_title('Intensity\nArtifact', fontsize=fs)
        axes[i, 4].set_title('Geometric\nDistortion', fontsize=fs)
        axes[i, 6].set_title('Resolution\n(with plastic)', fontsize=fs)
        axes[i, 8].set_title('Noise \n(Diff. Method)', fontsize=fs)

    plastic_image = images[0][slc]
    print('got plastic image @ RBW={:.3g}kHz'.format(rbw[0]))
    axes[i, 0].imshow(plastic_image, vmin=0, vmax=1, cmap='gray')

    axes[i, 0].set_ylabel(seqs2[i], fontsize=fs)
    
    metal_image = images[1][slc]
    axes[i, 1].imshow(metal_image, vmin=0, vmax=1, cmap='gray')

    intensity_map = maps_artifact[0][slc]
    im = axes[i, 2].imshow(intensity_map, vmin=-1, vmax=1, cmap='RdBu_r')
    if i == 0:
        cb = plt.colorbar(im, cax=axes[i, 3], ticks=[-1, 0, 1])
        cb.set_label(label='Relative Error', size=fs*0.75)
        axes[i, 3].yaxis.set_label_position('left')
        axes[i, 3].tick_params(labelsize=fs*0.75)
    artifacts_all += [maps_artifact[0].ravel()]
    mask_artifact = np.ones(artifacts_all[-1].shape, dtype=np.bool)
    # if i == 0:
    #     mask_artifact = (np.abs(artifacts_all[0]) > 0.3)
    # else:
    #     mask_artifact = np.logical_or(mask_artifact, np.abs(artifacts_all[0]) > 0.3)

    load_outputs(dirs[i], 'distortion')
    print('loaded distortion outputs for {} @ RBW={:.3g}kHz'.format(seqs[i], rbw[1]))
    result_mask = (results_masked[0][slc] != 0)
    measured_deformation = -deformation_fields[0][..., 0][slc]
    measured_deformation = measured_deformation * result_mask
    im = axes[i, 4].imshow(measured_deformation, vmin=-2, vmax=2, cmap='RdBu_r')
    if i == 0:
        cb = plt.colorbar(im, cax=axes[i, 5], ticks=[-2, -1, 0, 1, 2])
        cb.set_label(label='Displacement (pixels)', size=fs*0.75)
        axes[i, 5].yaxis.set_label_position('left')
        axes[i, 5].tick_params(labelsize=fs*0.75)
    x = np.linspace(0, 2, 50)
    y = np.abs(measured_deformation[np.abs(measured_deformation)>0].ravel())
    # axes[-1, 4].hist(np.abs(y), bins=np.linspace(0, 2, 21), alpha=0.5, label=seqs[i])
    density = stats.gaussian_kde(y)
    axes[-1, 4].plot(x, density(x), c=colors[i], label=seqs[i], linestyle=styles[i])
    axes[-1, 4].set_xticks([0, 1, 2])
    axes[-1, 4].tick_params(labelsize=fs*0.75)

    load_outputs(dirs[i], 'resolution')
    print('loaded resolution outputs')
    res_x = fwhms[0][..., 0][slc[:2] + (fwhms[0].shape[2] // 2 - 1,)]
    im = axes[i, 6].imshow(res_x, vmin=0, vmax=2, cmap='viridis')
    if i == 0:
        cb = plt.colorbar(im, cax=axes[i, 7], ticks=[0, 1, 2])
        cb.set_label(label='FWHM (pixels)', size=fs*0.75)
        axes[i, 7].yaxis.set_label_position('left')
        axes[i, 7].tick_params(labelsize=fs*0.75)
    x = np.linspace(0, 2, 50)
    y = res_x[res_x>0].ravel()
    # axes[-1, 6].hist(y, bins=np.linspace(0, 2, 21), alpha=0.5, label=seqs[i])
    density = stats.gaussian_kde(y)
    axes[-1, 6].plot(x, density(x), c=colors[i], label=seqs[i], linestyle=styles[i])
    axes[-1, 6].set_xlim([1, 2])
    axes[-1, 6].set_xticks([1, 1.5, 2])
    axes[-1, 6].tick_params(labelsize=fs*0.75)

    load_outputs(dirs[i], 'noise')
    print('loaded noise outputs for {} @ RBW={:.3g}kHz'.format(seqs[i], rbw[0]))
    plot_snr=False
    if plot_snr:
        snr = snrs[0][slc]
        im = axes[i, 8].imshow(snr, vmin=0, vmax=150, cmap='viridis')
        if i == 0:
            cb = plt.colorbar(im, cax=axes[i, 9], ticks=[0, 50, 100, 150])
            cb.set_label(label='SNR (a.u.)', size=fs*0.75)
            axes[i, 9].tick_params(labelsize=fs*0.75)
            axes[i, 9].yaxis.set_label_position('left')
        x = np.linspace(0, 10, 50)
        y = snr[snr>0].ravel() / np.sqrt(scan_times[i])
    else:
        # plot noise
        noise = noise_stds[0][slc] * np.sqrt(scan_times[i]) * 4
        im = axes[i, 8].imshow(noise, vmin=0, vmax=1, cmap='viridis')
        if i == 0:
            cb = plt.colorbar(im, cax=axes[i, 9], ticks=[0, 0.5, 1])
            cb.set_label(label=r'St. Dev. * $\sqrt{time}$ (a.u.)', size=fs*0.7)
            axes[i, 9].tick_params(labelsize=fs*0.75)
            axes[i, 9].yaxis.set_label_position('left')
        x = np.linspace(0, 1, 50)
        y = noise[noise>0].ravel()
    # axes[-1, 8].hist(y, bins=np.linspace(0, 1, 21), alpha=0.5, label=seqs[i])
    density = stats.gaussian_kde(y)
    axes[-1, 8].plot(x, density(x), c=colors[i], label=seqs[i], linestyle=styles[i])
    axes[-1, 8].tick_params(labelsize=fs*0.75)

for i in range(len(dirs)):
    x = np.linspace(0, 1, 50)
    y = np.abs(artifacts_all[i][mask_artifact].ravel())
    # axes[-1, 2].hist(y, bins=np.linspace(0, 1, 21), alpha=0.5, label=seqs[i])
    density = stats.gaussian_kde(y)
    axes[-1, 2].plot(x, density(x), c=colors[i], label=seqs[i][:-3], linestyle=styles[i])
    axes[-1, 2].set_xticks([0, 0.5, 1])
axes[-1, 2].legend(fontsize=fs*0.75)
axes[-1, 2].set_ylabel('Voxel Distribution', fontsize=fs)
axes[-1, 2].set_xlabel('Abs. Relative\nError', fontsize=fs)
axes[-1, 2].tick_params(labelsize=fs*0.75)
# axes[-1, 2].set_title('Inside Artifact Region')

# axes[-1, 4].legend()
# axes[-1, 4].set_ylabel('Voxels', fontsize=fs)
axes[-1, 4].set_xlabel('Abs. Displacement \n (pixels)', fontsize=fs)
# axes[-1, 4].set_title('Outside Artifact Region')

# axes[-1, 6].legend()
# axes[-1, 6].set_ylabel('Voxels', fontsize=fs)
axes[-1, 6].set_xlabel('FWHM\n(pixels)', fontsize=fs)
# axes[-1, 6].set_title('Entire Signal Region')

# axes[-1, 8].legend()
# axes[-1, 8].set_ylabel('Voxels', fontsize=fs)
axes[-1, 8].set_xlabel(r'St. Dev. * $\sqrt{time}$' + '\n(a.u.)', fontsize=fs)
# axes[-1, 8].set_title('Entire Signal Region')

plt.subplots_adjust(top=0.90, wspace=0.3)
plt.savefig(path.join(dirs[i], 'demo_comparison_{}_{}.png'.format(*short_seqs)), dpi=300)


for i in range(len(dirs)):
    load_outputs(dirs[i], 'artifact')
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 4))

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    metal_image = images[1][slc]
    axes[0].imshow(metal_image, vmin=0, vmax=1, cmap='gray')

    intensity_map = maps_artifact[0][slc]
    im = axes[1].imshow(intensity_map, vmin=-1, vmax=1, cmap='RdBu_r')

    axes[0].set_title('Metal', fontsize=fs)
    axes[1].set_title('Artifact', fontsize=fs)

    plt.tight_layout()
    plt.savefig(path.join(dirs[i], 'preview_{}.png'.format(short_seqs[i])), dpi=300)

plt.show()
