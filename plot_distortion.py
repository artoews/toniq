import numpy as np
import matplotlib.pyplot as plt
from os import path
import seaborn as sns
import scipy.ndimage as ndi
from skimage import morphology

import distortion
from plot import overlay_mask
from util import safe_divide, masked_copy


#### load data & some basic processing ####

root_dir = '/Users/artoews/root/code/projects/metal-phantom/abstract/root/'
slc = (slice(None), slice(35, 155), slice(65, 185), 30)
data = np.load(path.join(root_dir, 'distortion', 'outputs.npz'))
for var in data:
    globals()[var] = data[var]
fixed_image = images[1]
fixed_mask = masks_register[1]
fixed_image_masked = masked_copy(fixed_image, fixed_mask)
true_field = np.load(path.join(root_dir, 'field', 'field-metal.npy')) - np.load(path.join(root_dir, 'field', 'field-plastic.npy'))  # kHz
true_field = ndi.median_filter(true_field, footprint=morphology.ball(4))
# true_field = ndi.generic_filter(true_field, np.mean, footprint=morphology.ball(3))
# true_field = true_field[slc[1:]][slc2[1:]] * 1000  # Hz
true_field = true_field[slc[1:]] # kHz
num_trials = len(images) - 2


#### general plot settings ####

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

def letter_annotation(ax, xoffset, yoffset, letter):
    ax.text(xoffset, yoffset, letter, transform=ax.transAxes, size=18, weight='bold')


#### Initialize Sub-Figures ####

fig = plt.figure(figsize=(11, 8), layout='constrained')
fig_A, fig_BC = fig.subfigures(2, 1, hspace=0.1, height_ratios=[1, 1])
fig_B, fig_C = fig_BC.subfigures(1, 2, wspace=0.1, width_ratios = (2, 1))


#### Sub-Figure A ####

axes_A = fig_A.subplots(nrows=num_trials, ncols=5)
error_multiplier = 3
if num_trials == 1:
    axes_A = axes_A[None, :]
letter_annotation(axes_A[0][0], -0.2, 1.1, 'A')

def plot_image(ax, image, mask, xlabel=None, ylabel=None):
    im = ax.imshow(image, vmin=0, vmax=1, cmap='gray')
    overlay_mask(ax, mask)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_yticks([])
    return im

titles = ('Fixed Image',
          'Moving Image',
          'Registration',
          'Initial Error ({}x)'.format(error_multiplier),
          'Final Error ({}x)'.format(error_multiplier)
          )
for ax, title in zip(axes_A[0, :], titles):
    ax.set_title(title)

axes = axes_A
for i in range(num_trials):
    moving_mask = masks_register[1+i]
    moving_image_masked = masked_copy(images[2+i], moving_mask)
    init_error = np.abs(moving_image_masked - fixed_image_masked)
    result_error = np.abs(results_masked[i] - fixed_image_masked)
    init_mask =  (moving_image_masked != 0) * (fixed_image_masked != 0)
    result_mask = (results_masked[i] != 0)
    if i == 0:
        plot_image(axes[i, 0],
                   fixed_image_masked,
                   ~fixed_mask,
                   ylabel='RBW={:.3g}kHz'.format(rbw[0]))
    else:
        axes[i, 0].set_axis_off()
    plot_image(axes[i, 1],
               moving_image_masked,
               ~moving_mask,
               ylabel='RBW={:.3g}kHz'.format(rbw[1+i]))
    plot_image(axes[i, 2],
               results_masked[i],
               ~result_mask)
    plot_image(axes[i, 3],
               init_error * error_multiplier * init_mask,
               ~init_mask)
    im = plot_image(axes[i, 4],
               result_error * error_multiplier * result_mask,
               ~result_mask)

axes[1, 0].annotate("readout",
                    color='black',
                    xy=(0.5, 0.7),
                    xytext=(0.5, 0.1),
                    xycoords='axes fraction',
                    verticalalignment='bottom',
                    horizontalalignment='center',
                    arrowprops=dict(width=2, headwidth=8, headlength=8, color='black')
                    )
fig_A.colorbar(im, ax=axes, ticks=[0, 1], label='Pixel Intensity (a.u.)', location='right')


#### Sub-Figure B ####

axes_B = fig_B.subplots(nrows=num_trials, ncols=3)
if num_trials == 1:
    axes_B = axes_B[None, :]
letter_annotation(axes_B[0][0], -0.2, 1.1, 'B')

def plot_field(ax, image, mask, xlabel=None, ylabel=None):
    im = ax.imshow(image, vmin=-4, vmax=4, cmap='RdBu_r')
    overlay_mask(ax, mask)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_yticks([])
    return im

titles = ('Reference',
          'Registration',
          'Error'
          )
for ax, title in zip(axes_B[0, :], titles):
    ax.set_title(title)

axes = axes_B
for i in range(num_trials):
    if pbw[1+i] == pbw[0]:
        continue
    net_pbw = distortion.net_pixel_bandwidth(pbw[1+i], pbw[0])  # Hz
    result_mask = (results_masked[i] != 0)
    moving_image_masked = masked_copy(images[2+i], masks_register[1+i])
    simulated_deformation = true_field * 1000 / net_pbw
    measured_deformation = -deformation_fields[i][..., 0]
    plot_field(axes[i, 0],
               simulated_deformation * result_mask,
               ~result_mask,
               ylabel='RBW={:.3g}kHz'.format(rbw[1+i]))
    plot_field(axes[i, 1],
               measured_deformation * result_mask,
               ~result_mask)
    im = plot_field(axes[i, 2],
               (simulated_deformation - measured_deformation) * result_mask,
               ~result_mask)
fig_B.colorbar(im, ax=axes, ticks=[-4, -2, 0, 2, 4], label='Readout Displacement (pixels)', location='right')

#### Sub-Figure C ####

axes_C = fig_C.subplots()
letter_annotation(axes_C, -0.2, 1.1, 'C')
ax = axes_C
f_max = 1.5
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
styles = ['dotted', 'solid', 'dashed']
loosely_dashed = (0, (5, 10))
for i in range(num_trials):
    if pbw[1+i] == pbw[0]:
        continue
    result_mask = (results_masked[i] != 0)
    net_pbw = distortion.net_pixel_bandwidth(pbw[1+i], pbw[0]) / 1000 # kHz
    measured_deformation = -deformation_fields[i][..., 0]
    field_bins = np.round(true_field * 10) / 10
    sns.lineplot(x=(field_bins * result_mask).ravel(),
                 y=(measured_deformation * result_mask).ravel(),
                 ax=ax, legend='brief', label='RBW={0:.3g}kHz'.format(rbw[i+1]), color=colors[i], linestyle=styles[i])
    # ax.scatter((field_bins * result_mask).ravel(), (measured_deformation * result_mask).ravel(), c=colors[i], s=0.1, marker='.')
    ax.axline((-f_max, -f_max / net_pbw), (f_max, f_max / net_pbw), color=colors[i], linestyle=loosely_dashed)
    ax.set_xlim([-f_max, f_max])
    ax.set_ylim([-f_max / net_pbw, f_max / net_pbw])
ax.set_xlabel('Off-Resonance (kHz)')
ax.set_ylabel('Readout Displacement (pixels)')
plt.legend()
plt.grid()


#### save & show ####
plt.savefig(path.join(root_dir, 'distortion', 'distortion-summary.png'), dpi=300)
plt.show()