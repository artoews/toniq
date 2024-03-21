import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from os import path, makedirs

import gd
from plot import imshow2, label_panels, color_panels, label_encode_dirs, label_slice_pos
from plot_params import *

p = argparse.ArgumentParser(description='Make figure 7')
p.add_argument('save_dir', type=str, help='path where figure is saved')
p.add_argument('--out', type=str, default='out/mar20/mar4-fse125', help='path to main.py output folder')
p.add_argument('-y', '--y_slice', type=int, default=100, help='relative position of z slice (after crop); default=60')
p.add_argument('-z', '--z_slice', type=int, default=18, help='relative position of z slice (after crop); default=18')
p.add_argument('-e', '--error_scale', type=int, default=3, help='error map multiplier')
p.add_argument('-l', '--limit', type=int, default=2, help='distortion limit (pixels); default=2')
p.add_argument('-p', '--plot', action='store_true', help='show plots; default=2')

def plot_field_component(ax, gd_map, slc1, slc2, limit, mask, cmap):
    im, _, ax1, ax2 = imshow2(ax, gd_map, slc1, slc2, vmin=-limit, vmax=limit, mask=mask, cmap=cmap)
    cbar = gd.colorbar(ax, im, offset=0.35)
    cbar.set_label('Displacement (pixels)', size=SMALL_SIZE)
    return ax1, ax2

def plot_field_components(axes, gd_map, slc1, slc2, limit, mask, cmap=CMAP['distortion']):
    axes[0].set_title('Field X')
    axes[1].set_title('Field Y')
    axes[2].set_title('Field Z')
    ax1, ax2 = plot_field_component(axes[0], -gd_map[..., 0], slc1, slc2, limit, mask, cmap)
    label_encode_dirs(ax1)
    label_encode_dirs(ax2, x_label='z')
    plot_field_component(axes[1], gd_map[..., 1], slc1, slc2, limit, mask, cmap)
    plot_field_component(axes[2], gd_map[..., 2], slc1, slc2, limit, mask, cmap)

if __name__ == '__main__':

    args = p.parse_args()
    if not path.exists(args.save_dir):
        makedirs(args.save_dir)
    
    slc1 = (slice(None), slice(None), args.z_slice)
    slc2 = (slice(None), args.y_slice, slice(None))

    # plastic = np.load(path.join(args.out, 'gd-plastic-rigid-registered-masked.npy'))
    # plastic_mask = (plastic != 0)
    plastic = np.load(path.join(args.out, 'gd-plastic.npy'))
    plastic_mask = np.load(path.join(args.out, 'gd-plastic-mask.npy'))
    metal = np.load(path.join(args.out, 'gd-metal.npy'))
    metal_mask = np.load(path.join(args.out, 'gd-metal-mask.npy'))
    result = np.load(path.join(args.out, 'gd-plastic-registered-masked.npy'))
    gd_map = np.load(path.join(args.out, 'gd-map.npy'))
    ia_map = np.load(path.join(args.out, 'ia-map.npy'))
    init_mask = np.logical_and(plastic_mask, metal_mask)
    result_mask = np.logical_and(result != 0, metal_mask)

    fig = plt.figure(figsize=(FIG_WIDTH[2], FIG_WIDTH[2]*0.6))
    subfigs = fig.subfigures(1, 2, width_ratios=[2.25, 1], wspace=0.04)
    
    axes = subfigs[0].subplots(3, 3, gridspec_kw={'wspace': 0.03, 'hspace': 0.03, 'left': 0.03, 'right': 0.97, 'bottom': 0.03})
    axes[0, 0].set_title('Metal')
    axes[0, 1].set_title('Plastic')
    axes[0, 2].set_title('Plastic, Registered')
    if args.error_scale == 1:
        axes[1, 1].set_ylabel('Abs. Error')
        axes[2, 1].set_ylabel('Abs. Error,\nIA removed')
    else:
        axes[1, 1].set_ylabel('Abs. Error ({}x)'.format(args.error_scale))
        axes[2, 1].set_ylabel('Abs. Error ({}x),\nIA removed'.format(args.error_scale))
    _, _, ax1, ax2 = imshow2(axes[0, 0], metal, slc1, slc2, mask=~metal_mask)
    label_encode_dirs(ax1, buffer_text=True)
    label_encode_dirs(ax2, x_label='z', buffer_text=True)
    label_slice_pos(ax1, 1, slc2, slc1)
    label_slice_pos(ax2, -1, slc1, slc2)
    _, _, ax1, ax2 = imshow2(axes[0, 1], plastic, slc1, slc2, mask=~plastic_mask)
    # label_slice_pos(ax1, 1, slc2, slc1)
    # label_slice_pos(ax2, -1, slc1, slc2)
    _, _, ax1, ax2 = imshow2(axes[0, 2], result, slc1, slc2, mask=~result_mask)
    # label_slice_pos(ax1, 1, slc2, slc1)
    # label_slice_pos(ax2, -1, slc1, slc2)
    axes[1, 0].remove()
    imshow2(axes[1, 1], args.error_scale * np.abs(plastic - metal), slc1, slc2, mask=~init_mask)
    imshow2(axes[2, 1], args.error_scale * np.abs(plastic - metal / (1 + ia_map)), slc1, slc2, mask=~init_mask)
    axes[2, 0].remove()
    imshow2(axes[1, 2], args.error_scale * np.abs(result - metal), slc1, slc2, mask=~result_mask)
    imshow2(axes[2, 2], args.error_scale * np.abs(result - metal / (1 + ia_map)), slc1, slc2, mask=~result_mask)

    axes = subfigs[1].subplots(3, 1, gridspec_kw={'left': 0.03, 'right': 0.8, 'bottom': 0.03})
    plot_field_components(axes, -gd_map, slc1, slc2, args.limit, ~result_mask)

    # print('Max X distortion: ', np.max(np.abs(gd_map[..., 0])))
    # print('Max Y distortion: ', np.max(np.abs(gd_map[..., 1])))
    # print('Max Z distortion: ', np.max(np.abs(gd_map[..., 2])))

    label_panels(subfigs)
    color_panels(subfigs)

    plt.savefig(path.join(args.save_dir, 'figure7.png'), dpi=DPI, pad_inches=0)

    # ax = axes[1]
    # slope = 1/1.28
    # loosely_dashed = (0, (5, 10))
    # gx = -gd_map[..., 0][slc1]
    # gz = gd_map[..., 2][slc1]
    # mask = ~result_mask[slc1]
    # gx_bins = np.round(gx*50)/50
    # gz_bins = np.round(gz*50)/50
    # sns.lineplot(x=(gx_bins * mask).ravel(), y=(gz_bins * mask).ravel(), ax=ax)
    # ax.scatter((gx * mask).ravel(), (gz * mask).ravel(), s=0.1, marker='.')
    # ax.axline((-1, -slope), (1, slope), linestyle=loosely_dashed)
    # ax.set_xlabel('Off-Resonance (kHz)')
    # ax.set_ylabel('Displacement (pixels)')

    if args.plot:
        plt.show()