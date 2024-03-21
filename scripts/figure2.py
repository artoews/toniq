import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from os import path, makedirs

import ia, snr, gd, sr
from plot_params import *
from plot import remove_ticks, color_panels, label_panels, label_encode_dirs

def plot_inputs_panel(fig, up, um, sp, sm):
    kwargs = {'vmin': 0, 'vmax': 1, 'cmap': CMAP['image']}
    axes = fig.subplots(2, 2, gridspec_kw={'wspace': 0, 'hspace': 0})
    images = (up, um, sp, sm)
    for ax, im in zip(axes.flat, images):
        ax.imshow(im, **kwargs)
    axes[0, 0].set_title('Plastic')
    axes[0, 1].set_title('Metal')
    axes[0, 0].set_ylabel('Uniform')
    axes[1, 0].set_ylabel('Structured')
    # for ax in (axes[0, 1], axes[1, 0]):
    #     plt.text(0.77, 0.05, '2x', transform=ax.transAxes, color='white', fontsize=LARGE_SIZE) # , bbox=dict(facecolor='black', alpha=0.5))
    # axes[0, 0].annotate("readout",
    #     xy=(0.76, 0.30), xycoords='axes fraction',
    #     xytext=(0.76, 0.04), textcoords='axes fraction',
    #     horizontalalignment="center", size=SMALL_SIZE,
    #     arrowprops=dict(facecolor='black', width=0.5, headwidth=4, headlength=3)
    #     )
    label_encode_dirs(axes[0, 0])
    remove_ticks(axes)

def plot_output_panel(fig, input1, input2, map, mask, map_plotter, title):
    kwargs = {'vmin': 0, 'vmax': 1, 'cmap': CMAP['image']}
    gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.5)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1:3])
    ax1.imshow(input1, **kwargs)
    ax2.imshow(input2, **kwargs)
    ax3.set_title(title)
    map_plotter(ax3, map, mask)
    ax3.annotate("",
         xy=(-0.04, 0.5), xycoords='axes fraction',
         xytext=(-0.17, 0.5), textcoords='axes fraction',
         arrowprops=dict(facecolor='black', width=0.5, headwidth=4, headlength=3)
         )
    for ax in (ax1, ax2, ax3):
        remove_ticks(ax)
    return ax1, ax2, ax3

p = argparse.ArgumentParser(description='Make figure 2')
p.add_argument('save_dir', type=str, help='path where figure is saved')
p.add_argument('-r', '--root', type=str, default='out/mar20/mar4-fse125', help='root to demo data subfolder')
p.add_argument('-z', '--z_slice', type=int, default=19, help='z index of slice')
p.add_argument('-p', '--plot', action='store_true', help='show plots')

if __name__ == '__main__':

    args = p.parse_args()
    if not path.exists(args.save_dir):
        makedirs(args.save_dir)

    slc = (slice(None), slice(None), args.z_slice)

    implant_mask = np.load(path.join(args.root, 'implant-mask.npy'))
    ia_map = np.load(path.join(args.root, 'ia-map.npy'))
    gd_map = np.load(path.join(args.root, 'gd-map.npy'))
    gd_mask = np.load(path.join(args.root, 'gd-metal-registered-masked.npy')) != 0
    # gd_mask = np.load(path.join(args.root, 'gd-plastic-mask.npy'))
    snr_map = np.load(path.join(args.root, 'snr-map.npy'))
    snr_mask = np.load(path.join(args.root, 'snr-mask.npy'))
    res_map = np.load(path.join(args.root, 'fwhm-map.npy'))
    res_mask = np.load(path.join(args.root, 'res-mask.npy'))

    ia_plastic = np.load(path.join(args.root, 'ia-plastic.npy'))[slc]
    ia_metal = np.load(path.join(args.root, 'ia-metal.npy'))[slc]
    gd_plastic = np.load(path.join(args.root, 'gd-plastic.npy'))[slc]
    gd_metal = np.load(path.join(args.root, 'gd-metal.npy'))[slc]
    snr_image1 = np.load(path.join(args.root, 'snr-image-1.npy'))[slc]
    snr_image2 = np.load(path.join(args.root, 'snr-image-2.npy'))[slc]
    res_reference = np.load(path.join(args.root, 'res-image-ref.npy'))[slc]
    res_target = np.load(path.join(args.root, 'res-image-blurred.npy'))[slc]

    ia_map = ia_map[slc]
    gd_map = -gd_map[..., 0][slc]
    gd_mask = gd_mask[slc]
    snr_map = snr_map[slc]
    snr_mask = snr_mask[slc]
    res_map = res_map[..., 0][slc]
    res_mask = (res_map != 0)

    fig = plt.figure(figsize=(FIG_WIDTH[2], FIG_WIDTH[2]*0.35))
    margin = 0.04
    subfigs = fig.subfigures(1, 2, width_ratios=[1, 2], wspace=margin/2, hspace=margin)
    subsubfigs = subfigs[1].subfigures(2, 2, wspace=margin/2, hspace=margin)

    plot_inputs_panel(subfigs[0], ia_plastic, ia_metal, gd_plastic, gd_metal)
    plot_output_panel(subsubfigs[0, 0], ia_plastic, ia_metal, ia_map, None, ia.plot_map, 'Intensity Artifact')
    plot_output_panel(subsubfigs[0, 1], snr_image1, snr_image2, snr_map, snr_mask, snr.plot_map, 'SNR')
    plot_output_panel(subsubfigs[1, 0], gd_plastic, gd_metal, gd_map, gd_mask, gd.plot_map, 'Geometric Distortion')
    # ax, _, _ = plot_output_panel(subsubfigs[1, 1], res_reference, res_target, res_map, res_mask, plot_res_map, 'Spatial Resolution')
    ax, _, _ = plot_output_panel(subsubfigs[1, 1], res_reference, res_target, res_map, res_map != 0, sr.plot_map, 'Spatial Resolution')

    for spine in ax.spines.values():
        spine.set_edgecolor('blue')

    color_panels([subfigs[0],] + list(subsubfigs.ravel()))
    label_panels([subfigs[0],] + list(subsubfigs.ravel()))

    plt.savefig(path.join(args.save_dir, 'figure2.png'), dpi=DPI)

    if args.plot:
        plt.show()
