import matplotlib.pyplot as plt
import numpy as np
from os import path
import scipy.stats as stats

from plot import plotVolumes

def load_outputs(root, subfolder):
    data = np.load(path.join(root, subfolder, 'outputs.npz'))
    for var in data:
        globals()[var] = data[var]

## general plot settings
SMALL_SIZE = 10
MEDIUM_SIZE = 12
LARGE_SIZE = 14
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=LARGE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the x tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the y tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)   # fontsize of the figure title
plt.rc('lines', linewidth=1.0)
styles = ['dotted', 'solid', 'dashed']

## imshow keyword arguments
image_kwargs = {'vmin': 0, 'vmax': 1, 'cmap': 'gray'}
artifact_kwargs = {'vmin': -1, 'vmax': 1, 'cmap': 'RdBu_r'}
distortion_kwargs = {'vmin': -2, 'vmax': 2, 'cmap': 'RdBu_r'}
resolution_kwargs = {'vmin': 1, 'vmax': 3, 'cmap': 'viridis'}
noise_kwargs = {'vmin': 0, 'vmax': 1, 'cmap': 'viridis'}

## scan time reference
# FSE @ [250, 125, 100] kHz = 2:20 (140s)
# MSL @ [250, 125, 100] kHz = 7:42 (462s)

## data directories
fse_dir = '/Users/artoews/root/code/projects/metal-phantom/bmr/demo-fse-250'
msl_dir = ['/Users/artoews/root/code/projects/metal-phantom/bmr/demo-msl-250',
           '/Users/artoews/root/code/projects/metal-phantom/bmr/demo-msl-100']
seq_names = ['FSE', 'MAVRIC-SL', 'MAVRIC-SL']
scan_times = [140, 462, 462]  # seconds
dirs = [fse_dir] + msl_dir

## slice to show
slc = (slice(None), slice(None), 15)

## figure setup 
fig, axs = plt.subplots(nrows=len(dirs)+1, ncols=6, figsize=(13, 8), layout='constrained')
axs[-1, 0].set_axis_off()
axs[-1, 1].set_axis_off()
for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

## column titles
titles = ('Plastic',
          'Metal',
          'Intensity\nArtifact',
          'Geometric\nDistortion',
          'Resolution\n(with plastic)',
          'Noise\n(Diff. Method)'
          )
for ax, title in zip(axs[0, :], titles):
    ax.set_title(title)

## artifact column
for i in range(len(dirs)):
    load_outputs(dirs[i], 'artifact')
    rbw_i = rbw[0]
    plastic_image = images[0]
    metal_image = images[1]
    intensity_map = maps_artifact[0]
    print('loaded artifact outputs for {} @ RBW={:.3g}kHz'.format(seq_names[i], rbw_i))
    print('got plastic image @ RBW={:.3g}kHz'.format(rbw_i))
    axs[i, 0].set_ylabel('{}\nRBW={:.3g}kHz'.format(seq_names[i], rbw_i), fontsize=LARGE_SIZE)
    axs[i, 0].imshow(plastic_image[slc], **image_kwargs)
    axs[i, 1].imshow(metal_image[slc], **image_kwargs)
    im = axs[i, 2].imshow(intensity_map[slc], **artifact_kwargs)
    x = np.linspace(0, 1, 50)
    y = np.abs(intensity_map.ravel())
    density = stats.gaussian_kde(y)
    axs[-1, 2].plot(x, density(x),
                    label='{} {:.3g}'.format(seq_names[i][:3], rbw_i),
                    linestyle=styles[i])
axs[-1, 2].set_ylabel('Voxel Distribution')
axs[-1, 2].legend()
axs[-1, 2].set_xticks([0, 0.5, 1])
axs[-1, 2].set_xlabel('Abs. Relative\nError')
cb = fig.colorbar(im, ax=axs[0, 2], ticks=[-1, 0, 1], label='Relative Error')
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
for i in range(len(dirs)):
    load_outputs(dirs[i], 'distortion')
    rbw_i = rbw[1]
    # results_masked_i = results_masked[0]
    results_masked_i = results[0]
    print('deformation fields', deformation_fields.shape)
    print('results masked', results_masked_i.shape)
    deformation_fields_i = deformation_fields[0]
    print('loaded distortion outputs for {} @ RBW={:.3g}kHz'.format(seq_names[i], rbw_i))
    result_mask = (results_masked_i != 0)
    measured_deformation = -deformation_fields_i[..., 0]
    measured_deformation = measured_deformation * result_mask
    im = axs[i, 3].imshow(measured_deformation[slc], **distortion_kwargs)
    x = np.linspace(0, 2, 50)
    y = np.abs(measured_deformation.ravel())
    density = stats.gaussian_kde(y)
    axs[-1, 3].plot(x, density(x), linestyle=styles[i])
axs[-1, 3].set_xticks([0, 1, 2])
axs[-1, 3].set_xlabel('Abs. Displacement\n(pixels)')
axs[-1, 3].set_ylim([0, 1])
cb = fig.colorbar(im, ax=axs[0, 3], ticks=[-2, -1, 0, 1, 2], label='Displacement (pixels)')


## resolution column
for i in range(len(dirs)):
    load_outputs(dirs[i], 'resolution')
    print('loaded resolution outputs')
    res_x = fwhms[0][..., 0]
    im = axs[i, 4].imshow(res_x[slc], **resolution_kwargs)
    x = np.linspace(0, 3, 50)
    y = res_x[res_x>0].ravel()
    density = stats.gaussian_kde(y)
    axs[-1, 4].plot(x, density(x), linestyle=styles[i])
axs[-1, 4].set_xlim([1, 3])
axs[-1, 4].set_xticks([1, 2, 3])
axs[-1, 4].set_xlabel('FWHM\n(mm)')
cb = fig.colorbar(im, ax=axs[0, 4], ticks=[1, 2, 3], label='FWHM (mm)')

## noise column
for i in range(len(dirs)):
    load_outputs(dirs[i], 'noise')
    rbw_i = rbw[0]
    noise_stds_i = noise_stds[0]
    print('loaded noise outputs for {} @ RBW={:.3g}kHz'.format(seq_names[i], rbw_i))
    noise = noise_stds_i * np.sqrt(scan_times[i]) * 4
    im = axs[i, 5].imshow(noise[slc], **noise_kwargs)
    x = np.linspace(0, 1, 50)
    y = noise[noise>0].ravel()
    density = stats.gaussian_kde(y)
    axs[-1, 5].plot(x, density(x), linestyle=styles[i])
axs[-1, 5].set_xticks([0, 0.5, 1])
axs[-1, 5].set_xlabel(r'St. Dev. * $\sqrt{time}$' + '\n(a.u.)')
cb = fig.colorbar(im, ax=axs[0, 5], ticks=[0, 1], label=r'St. Dev. * $\sqrt{time}$ (a.u.)')

## save & show
plt.savefig(path.join(dirs[i], 'demo_revised.png'), dpi=300)
plt.show()
