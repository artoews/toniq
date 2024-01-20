import numpy as np
import matplotlib.pyplot as plt
from os import path
import seaborn as sns
import scipy.ndimage as ndi
from skimage import morphology

from distortion import net_pixel_bandwidth, simulated_deformation_fse
from plot import overlay_mask, letter_annotation, imshow2
from util import masked_copy

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



def plot_image(ax, image, mask, slc1, slc2, xlabel=None, ylabel=None):
    # im = ax.imshow(image, vmin=0, vmax=1, cmap='gray')
    im, _ = imshow2(ax, image, slc1, slc2, mask=mask)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_yticks([])
    return im

def plot_image_results(fig, masks, images, results, rbw):
    slc_xy = (slice(None), slice(None), images[0].shape[2] // 2)
    slc_xz = (slice(None), images[0].shape[1] // 2, slice(None))
    num_trials = len(results)
    axes = fig.subplots(nrows=2*num_trials, ncols=3)
    error_multiplier = 2

    titles = ('Plastic', 'Metal', 'Registration')
    for ax, title in zip(axes[0, :], titles):
        ax.set_title(title)
    
    for i in range(num_trials):
        axes[2*i+1, 1].set_ylabel('Error ({}x)'.format(error_multiplier))
        axes[2*i+1, 0].set_axis_off()
        fixed_mask = masks[2*i]
        moving_mask = masks[2*i+1]
        fixed_image_masked = masked_copy(images[2*i], fixed_mask)
        moving_image_masked = masked_copy(images[2*i+1], moving_mask)
        init_error = np.abs(moving_image_masked - fixed_image_masked)
        result_error = np.abs(results[i] - fixed_image_masked)
        init_mask = np.logical_and(moving_image_masked, fixed_image_masked)
        result_mask = np.logical_and(results[i] != 0, fixed_image_masked)
        plot_image(axes[2*i, 0],
                   fixed_image_masked,
                   ~fixed_mask,
                   slc_xy,
                   slc_xz,
                   ylabel='RBW={:.3g}kHz'.format(rbw[i]))
        plot_image(axes[2*i, 1],
                   moving_image_masked,
                   ~moving_mask,
                   slc_xy,
                   slc_xz)
        plot_image(axes[2*i, 2],
                   results[i],
                   ~result_mask,
                   slc_xy,
                   slc_xz)
        plot_image(axes[2*i+1, 1],
                   init_error * error_multiplier * init_mask,
                   ~init_mask,
                   slc_xy,
                   slc_xz)
        im = plot_image(axes[2*i+1, 2],
                   result_error * error_multiplier * result_mask,
                   ~result_mask,
                   slc_xy,
                   slc_xz)
        fig.colorbar(im, ax=axes[2*i:2*i+2, :], ticks=[0, 1], label='Pixel Intensity (a.u.)', location='right')

    if num_trials > 1:
        axes[1, 0].annotate("readout",
                            color='black',
                            xy=(0.5, 0.7),
                            xytext=(0.5, 0.1),
                            xycoords='axes fraction',
                            verticalalignment='bottom',
                            horizontalalignment='center',
                            arrowprops=dict(width=2, headwidth=8, headlength=8, color='black')
                            )
    return axes

def plot_field(ax, image, mask, slc1, slc2, xlabel=None, ylabel=None):
    # im = ax.imshow(image, vmin=-4, vmax=4, cmap='RdBu_r')
    im, _ = imshow2(ax, image, slc1, slc2, vmin=-4, vmax=4, cmap='RdBu_r', mask=mask)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_yticks([])
    return im

def plot_field_results(fig, results, true_field, deformation_fields, rbw, pbw, field_dir=0):
    slc_xy = (slice(None), slice(None), results[0].shape[2] // 2)
    slc_xz = (slice(None), results[0].shape[1] // 2, slice(None))
    axes = fig.subplots(nrows=len(results), ncols=3)
    num_trials = len(results)
    if num_trials == 1:
        axes = axes[None, :] 

    titles = ('Simulation', 'Registration', 'Difference')
    for ax, title in zip(axes[0, :], titles):
        ax.set_title(title)

    # TODO pass these in
    gx = [1.912, 0.956, 0.478] # G/cm
    gz = 1.499 # G/cm
    for i in range(num_trials):
        net_pbw = pbw[i] # assumes registration's fixed image was plastic, so no distortion
        # net_pbw = net_pixel_bandwidth(pbw[1+i], pbw[0])  # Hz
        result_mask = (results[i] != 0)
        # simulated_deformation = true_field * 1000 / net_pbw
        simulated_deformation = simulated_deformation_fse(true_field, gx[i], gz, 1.2, 1.2, pbw_kHz=net_pbw / 1000)
        plot_field(axes[i, 0],
            simulated_deformation[..., field_dir] * result_mask,
            ~result_mask,
            slc_xy,
            slc_xz,
            ylabel='RBW={:.3g}kHz'.format(rbw[i]))
        measured_deformation = deformation_fields[i][..., field_dir]
        if field_dir == 0:
            measured_deformation = -measured_deformation
        plot_field(axes[i, 1],
            measured_deformation * result_mask,
            ~result_mask,
            slc_xy,
            slc_xz)
        im = plot_field(axes[i, 2],
            (simulated_deformation[..., field_dir] - measured_deformation) * result_mask,
            ~result_mask,
            slc_xy,
            slc_xz)
    fig.colorbar(im, ax=axes, ticks=[-4, -2, 0, 2, 4], label='Displacement (pixels)', location='right')
    return axes

def plot_summary_results(fig, results, reference, field, rbw, pbw):
    axes = fig.subplots()
    f_max = 1.5
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    styles = ['dotted', 'solid', 'dashed']
    loosely_dashed = (0, (5, 10))
    for i in range(len(results)):
        net_pbw = pbw[i] / 1000 # assumes i=0 is plastic
        # net_pbw = net_pixel_bandwidth(pbw[i], pbw[0]) / 1000 # kHz
        result_mask = (results[i] != 0)
        field_bins = np.round(reference * 10) / 10
        sns.lineplot(x=(field_bins * result_mask).ravel(),
                     y=(field[i] * result_mask).ravel(),
                     ax=axes, legend='brief', label='RBW={0:.3g}kHz'.format(rbw[i]), color=colors[i], linestyle=styles[i])
        # ax.scatter((field_bins * result_mask).ravel(), (measured_deformation * result_mask).ravel(), c=colors[i], s=0.1, marker='.')
        axes.axline((-f_max, -f_max / net_pbw), (f_max, f_max / net_pbw), color=colors[i], linestyle=loosely_dashed)
        axes.set_xlim([-f_max, f_max])
        axes.set_ylim([-4, 4])
    axes.set_xlabel('Off-Resonance (kHz)')
    axes.set_ylabel('Displacement (pixels)')
    plt.legend()
    plt.grid()
    return axes

if __name__ == '__main__':

    # Load & Pre-processing
    root_dir = '/Users/artoews/root/code/projects/metal-phantom/tmp/'
    slc = (slice(35, 155), slice(65, 185), 30)
    data = np.load(path.join(root_dir, 'distortion', 'outputs.npz'))
    for var in data:
        globals()[var] = data[var]
    true_field_kHz = np.load(path.join(root_dir, 'field', 'field_diff_Hz.npy'))[slc] / 1000

    ## Setup
    fig = plt.figure(figsize=(11, 8), layout='constrained')
    fig_A, fig_BC = fig.subfigures(2, 1, hspace=0.1, height_ratios=[1, 1])
    fig_B, fig_C = fig_BC.subfigures(1, 2, wspace=0.1, width_ratios = (2, 1))

    ## Plot
    axes_A = plot_image_results(fig_A, masks_register, images, results, rbw)
    letter_annotation(axes_A[0][0], -0.2, 1.1, 'A')
    axes_B = plot_field_results(fig_B, results, true_field_kHz, deformation_fields, rbw, pbw)
    letter_annotation(axes_B[0][0], -0.2, 1.1, 'B')
    axes_C = plot_summary_results(fig_C, results, true_field_kHz, deformation_fields, rbw, pbw)
    letter_annotation(axes_C, -0.2, 1.1, 'C')

    ## Save
    plt.savefig(path.join(root_dir, 'distortion', 'distortion-summary.png'), dpi=300)
    plt.show()
