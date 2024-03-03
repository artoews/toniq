import matplotlib.pyplot as plt
import numpy as np
from os import path
import scipy.stats as stats

from plot import overlay_mask
from plot_params import *

def load_outputs(root, subfolder):
    data = np.load(path.join(root, subfolder, 'outputs.npz'))
    for var in data:
        globals()[var] = data[var]

styles = ['dotted', 'solid', 'dashed']

## imshow keyword arguments
image_kwargs = {'vmin': 0, 'vmax': 1, 'cmap': CMAP['image']}
artifact_kwargs = {'vmin': -1, 'vmax': 1, 'cmap': CMAP['artifact']}
distortion_kwargs = {'vmin': -3, 'vmax': 3, 'cmap': CMAP['distortion']}
resolution_kwargs = {'vmin': 1.5, 'vmax': 3, 'cmap': CMAP['resolution']}
snr_kwargs = {'vmin': 0, 'vmax': 10, 'cmap': CMAP['snr']}

## scan time reference
# FSE @ [250, 125] kHz = 2:30 (150s)
# MSL @ [250, 125] kHz = 5:09 (309s)

## data directories
fse_dir = '/Users/artoews/root/code/projects/metal-phantom/bmr/demo-fse-250'
msl_dir = ['/Users/artoews/root/code/projects/metal-phantom/bmr/demo-msl-250',
           '/Users/artoews/root/code/projects/metal-phantom/bmr/demo-msl-100']
seq_names = ['FSE', 'MAVRIC-SL', 'MAVRIC-SL']
scan_times = [140, 462, 462]  # seconds Jan 15
scan_times = [150, 309, 309]  # seconds Jan 21
dirs = [fse_dir] + msl_dir

## slice to show
slc = (slice(None), slice(None), 19)

## figure setup 
fig, axs = plt.subplots(nrows=len(dirs), ncols=5, figsize=(12, 8), layout='constrained')
axs[-1, 0].set_axis_off()
for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

## column titles
titles = (
          'Metal',
          'Intensity\nArtifact',
          'Readout\nDistortion',
          'Readout\nResolution',
          'SNR Efficiency'
          )
for ax, title in zip(axs[0, :], titles):
    ax.set_title(title)

## artifact column
signal_ref_list = []
intensity_maps = []
for i in range(len(dirs)-1):
    load_outputs(dirs[i+1], 'artifact')
    rbw_i = rbw[0]
    plastic_image = images[0]
    signal_ref_list.append(signal_refs[0])
    metal_image = images[1]
    intensity_map = maps_artifact[0]
    intensity_maps.append(intensity_map)
    print('loaded artifact outputs for {} @ RBW={:.3g}kHz'.format(seq_names[i+1], rbw_i))
    print('got plastic image @ RBW={:.3g}kHz'.format(rbw_i))
    axs[i, 0].set_ylabel('{}\nRBW={:.3g}kHz'.format(seq_names[i+1], rbw_i), fontsize=LARGE_SIZE)
    # axs[i, 0].imshow(plastic_image[slc], **image_kwargs)
    axs[i, 0].imshow(metal_image[slc], **image_kwargs)
    im = axs[i, 1].imshow(intensity_map[slc], **artifact_kwargs)
    x = np.linspace(0, 1, 50)
    y = np.abs(intensity_map.ravel())
    density = stats.gaussian_kde(y)
    axs[-1, 1].plot(x, density(x),
                    label='{} {:.3g}'.format(seq_names[i][:3], rbw_i),
                    linestyle=styles[i])
axs[-1, 1].set_ylabel('Voxel Distribution')
axs[-1, 1].legend()
axs[-1, 1].set_xticks([0, 0.5, 1])
axs[-1, 1].set_xlabel('Abs. Relative Error')
cb = fig.colorbar(im, ax=axs[-2, 1], ticks=[-1, 0, 1], label='Relative Error', location='bottom')
axs[-1, 0].annotate("readout",
                    color='black',
                    xy=(0.5, 0.7),
                    xytext=(0.5, 0.1),
                    xycoords='axes fraction',
                    verticalalignment='bottom',
                    horizontalalignment='center',
                    arrowprops=dict(width=2, headwidth=8, headlength=8, color='black')
                    )

## distortion column
resolution_mm = 1.2
mask_register_list = []
for i in range(len(dirs)-1):
    load_outputs(dirs[i+1], 'distortion')
    rbw_i = rbw[0]
    # results_masked_i = results_masked[0]
    mask_register_list.append(masks_register[1])
    results_masked_i = results[0]
    deformation_fields_i = deformation_fields[0]
    print('loaded distortion outputs for {} @ RBW={:.3g}kHz'.format(seq_names[i+1], rbw_i))
    result_mask = (results_masked_i != 0)
    measured_deformation = -deformation_fields_i[..., 0] * resolution_mm
    measured_deformation = measured_deformation * result_mask
    im = axs[i, 2].imshow(measured_deformation[slc], **distortion_kwargs)
    x = np.linspace(0, 3, 50)
    y = np.abs(measured_deformation[mask_register_list[i]].ravel())
    density = stats.gaussian_kde(y)
    axs[-1, 2].plot(x, density(x), linestyle=styles[i])
    overlay_mask(axs[i, 2], ~mask_register_list[i][slc])
axs[-1, 2].set_xticks([0, 1, 2, 3])
axs[-1, 2].set_xlabel('Abs. Displacement\n(mm)')
axs[-1, 2].set_ylim([0, 2])
cb = fig.colorbar(im, ax=axs[-2, 2], ticks=[-3, 0, 3], label='Displacement (mm)', location='bottom')


## resolution column
for i in range(len(dirs)-1):
    load_outputs(dirs[i+1], 'resolution')
    print('loaded resolution outputs')
    res_x = fwhms[0][..., 0]
    im = axs[i, 3].imshow(res_x[slc], **resolution_kwargs)
    x = np.linspace(0, 3, 50)
    y = res_x[res_x>0].ravel()
    density = stats.gaussian_kde(y)
    axs[-1, 3].plot(x, density(x), linestyle=styles[i])
axs[-1, 3].set_xlim([1.5, 3])
axs[-1, 3].set_xticks([1.5, 2, 2.5, 3])
axs[-1, 3].set_xlabel('FWHM\n(mm)')
cb = fig.colorbar(im, ax=axs[-2, 3], ticks=[1.5, 3], label='FWHM (mm)', location='bottom')

## noise column
for i in range(len(dirs)-1):
    load_outputs(dirs[i+1], 'snr')
    rbw_i = rbw[0]
    noise_stds_i = noise_stds[0]
    # snr_i = snrs[0]
    snr_i = signal_ref_list[i] / noise_stds_i
    print('loaded snr outputs for {} @ RBW={:.3g}kHz'.format(seq_names[i+1], rbw_i))
    noise = noise_stds_i * np.sqrt(scan_times[i+1]) * 4
    snr = snr_i / np.sqrt(scan_times[i+1])
    # im = axs[i, 4].imshow(noise[slc], **noise_kwargs)
    im = axs[i, 4].imshow(snr[slc], **snr_kwargs)
    # mask = np.abs(intensity_maps[i][slc]) > 0.2
    overlay_mask(axs[i, 4], ~mask_register_list[i][slc])
    # x = np.linspace(0, 1, 50)
    # y = noise[noise>0].ravel()
    x = np.linspace(0, 10, 50)
    y = snr[mask_register_list[i]].ravel()
    density = stats.gaussian_kde(y)
    axs[-1, 4].plot(x, density(x), linestyle=styles[i])
# axs[-1, 4].set_xticks([0, 0.5, 1])
# axs[-1, 4].set_xlabel(r'St. Dev. * $\sqrt{time}$' + '\n(a.u.)')
# cb = fig.colorbar(im, ax=axs[0, 4], ticks=[0, 1], label=r'St. Dev. * $\sqrt{time}$ (a.u.)')
axs[-1, 4].set_xticks([0, 5, 10])
axs[-1, 4].set_xlabel(r'SNR / $\sqrt{time}$' + '\n' + r'(s$^{-1/2}$)')
cb = fig.colorbar(im, ax=axs[-2, 4], ticks=[0, 5, 10], label=r'SNR / $\sqrt{time}$ (s$^{-1/2}$)', location='bottom')

## save & show
plt.savefig(path.join(dirs[i], 'demo_revised.png'), dpi=300)
plt.show()