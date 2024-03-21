import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.stats as stats

from os import path, makedirs

import ia, snr, sr
from plot import remove_ticks, label_panels, color_panels, label_encode_dirs
from plot_params import *
from util import safe_divide

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

protocol_names = ['1 kHz/pixel', '0.5 kHz/pixel']

def plot_row_images(axes, image1, image2, slc):
    axes[0].imshow(image1[slc], vmin=0, vmax=1, cmap=CMAP['image'])
    im = axes[1].imshow(image2[slc], vmin=0, vmax=1, cmap=CMAP['image'])
    axes[2].remove()
    axes[3].remove()
    axes[4].remove()
    label_encode_dirs(axes[0])
    # cbar = plt.colorbar(im, cax=colorbar_axis(axes[1], offset=0), ticks=[0, 1])
    # cbar.set_label('Intensity', size=SMALL_SIZE)
    # cbar.ax.tick_params(labelsize=SMALLER_SIZE)

def plot_row_ia(axes, map1, map2, slc):
    ia.plot_map(axes[0], map1[slc], None, show_cbar=False)
    ia.plot_map(axes[1], map2[slc], None)
    cbar = ia.plot_map(axes[3], np.abs(map2[slc]) - np.abs(map1[slc]), None, lim=0.3)
    cbar.set_label('Difference of\nAbs. Relative Error', size=SMALL_SIZE)
    axes[2].remove()
    axes[4].remove()

def plot_row_snr(axes, map1, map2, mask1, mask2, slc):
    snr.plot_map(axes[0], map1[slc], mask1[slc], show_cbar=False)
    snr.plot_map(axes[1], map2[slc], mask2[slc])
    cbar = snr.plot_map(axes[3], safe_divide(map2[slc], map1[slc]), np.logical_and(mask1[slc], mask2[slc]), ticks=[0, 1.41, 2.82])
    cbar.set_label('Ratio of SNR', size=SMALL_SIZE)
    axes[2].remove()
    axes[4].remove()

def plot_row_res(axes, map1, map2, mask1, mask2, slc):
    sr.plot_map(axes[0], map1[slc], mask1[slc], show_cbar=False)
    sr.plot_map(axes[1], map2[slc], mask2[slc])
    # plot_res_map(axes[3], map2[slc] - map1[slc], None, vmin=0, vmax=1)
    cbar = sr.plot_map(axes[3], safe_divide(map2[slc], map1[slc]), np.logical_and(mask1[slc], mask2[slc]), vmin=1, vmax=2)
    cbar.set_label('Ratio of FWHM', size=SMALL_SIZE)
    axes[2].remove()
    axes[4].remove()

def plot_summary(ax, map1, map2, pts, slc1, slc2, thresh=None):
    ax.set_yticks([])
    styles = ['solid', 'dashed', 'dotted']
    labels = []
    lines = []
    for map, color, name in zip((map1, map2), colors, protocol_names):
        plot_distribution(ax, map, pts, styles[0], color, thresh)
        plot_distribution(ax, map[slc1], pts, styles[1], color, thresh)
        labels += [name + ', volume', name + ', slice']
        lines += [Line2D([0], [0], color=color, linestyle=styles[0]), 
                  Line2D([0], [0], color=color, linestyle=styles[1])]
        # plot_distribution(ax, map[slc2], pts, styles[2], color, thresh)
        # labels += [name + ', volume', name + ', slice', name + ', ROI']
        # lines += [Line2D([0], [0], color=color, linestyle=styles[0]), 
        #           Line2D([0], [0], color=color, linestyle=styles[1]),
        #           Line2D([0], [0], color=color, linestyle=styles[2])]
    return lines, labels

def plot_distribution(ax, data, pts, line_style, line_color, thresh):
    if thresh is not None:
        data = data[data > thresh]
    density = stats.gaussian_kde(data.ravel())
    ax.plot(pts, density(pts), linestyle=line_style, color=line_color)

def make_plot_white(ax):
    remove_ticks(ax)
    ax.xaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    for child in ax.get_children():
        if isinstance(child, mpl.spines.Spine):
            child.set_color('white')

p = argparse.ArgumentParser(description='Make figure 10')
p.add_argument('save_dir', type=str, help='path where figure is saved')
p.add_argument('--root1', type=str, default='out/mar20/mar4-msl125', help='root to demo data subfolder 1')
p.add_argument('--root2', type=str, default='out/mar20/mar4-msl63', help='root to demo data subfolder 2')
p.add_argument('-z', '--z_slice', type=int, default=18, help='relative position of z slice (after crop); default=18')
p.add_argument('-p', '--plot', action='store_true', help='show plots')


if __name__ == '__main__':

    args = p.parse_args()
    if not path.exists(args.save_dir):
        makedirs(args.save_dir)
    
    slc = (slice(None), slice(None), args.z_slice)
    roi = (slice(100, 110), slice(60, 70), args.z_slice)
    # roi = (slice(20, 100), slice(20, 100), slice(13, 23))
    offset = 7
    roi_hr = (slice(roi[0].start-offset, roi[0].stop-offset),
              slice(roi[1].start-offset, roi[1].stop-offset),
              roi[2])

    image1 = np.load(path.join(args.root1, 'gd-metal.npy'))
    image2 = np.load(path.join(args.root2, 'gd-metal.npy'))
    ia1 = np.load(path.join(args.root1, 'ia-map.npy'))
    ia2 = np.load(path.join(args.root2, 'ia-map.npy'))
    snr1 = np.load(path.join(args.root1, 'snr-map.npy'))
    snr2 = np.load(path.join(args.root2, 'snr-map.npy'))
    snr1_mask = np.load(path.join(args.root1, 'snr-mask.npy'))
    snr2_mask = np.load(path.join(args.root2, 'snr-mask.npy'))
    res1 = np.load(path.join(args.root1, 'fwhm-map.npy'))[..., 0]
    res2 = np.load(path.join(args.root2, 'fwhm-map.npy'))[..., 0]
    # res1_mask = np.load(path.join(args.root1, 'res-mask.npy'))
    # res2_mask = np.load(path.join(args.root2, 'res-mask.npy'))


    fig = plt.figure(figsize=(FIG_WIDTH[2], FIG_WIDTH[2]*0.75))
    subfigs = fig.subfigures(1, 2, hspace=0.03, width_ratios=[3, 1], wspace=0.02)
    axes = subfigs[0].subplots(nrows=4, ncols=5, width_ratios=[1, 1, 0.75, 1, 0.5], 
                               gridspec_kw={'wspace': 0.05, 'hspace': 0.15, 'left': 0.14, 'right': 0.97, 'bottom': 0.05, 'top': 0.92})
    remove_ticks(axes)
    plot_row_images(axes[0, :], image1, image2, slc)
    plot_row_ia(axes[1, :], ia1, ia2, slc)
    plot_row_snr(axes[2, :], snr1, snr2, snr1_mask, snr2_mask, slc)
    plot_row_res(axes[3, :], res1, res2, res1!=0, res2!=0, slc)
    axes[0, 0].set_title('1 kHz/pixel')
    axes[0, 1].set_title('0.5 kHz/pixel')
    # axes[1, 3].set_title('Comparison')
    for ax, label in zip(axes[:, 0], ('Metal\nImage', 'Intensity\nArtifact', 'SNR', 'Spatial\nResolution')):
        ax.set_ylabel(label, rotation='horizontal', va='center', ha='center', labelpad=25)

    axes = subfigs[1].subplots(nrows=4, ncols=1, height_ratios=[1, 1, 1, 1],
                               gridspec_kw={'hspace': 0.8, 'left': 0.1, 'right': 0.9, 'bottom': 0.1, 'top': 0.92})
    plot_summary(axes[1], ia1, ia2, np.linspace(0, 1, 50), slc, roi)
    plot_summary(axes[2], snr1, snr2, np.linspace(0, 200, 50), slc, roi, thresh=0)
    lines, labels = plot_summary(axes[3], res1, res2, np.linspace(1, 4, 50), slc, roi_hr, thresh=0)
    make_plot_white(axes[0])
    axes[0].legend(lines, labels, loc='center', borderpad=1)
    axes[1].set_xlabel('Abs. Relative Error')
    axes[2].set_xlabel('SNR')
    axes[3].set_xlabel('FWHM (mm)')
    
    label_panels(subfigs)
    color_panels(subfigs)

    plt.savefig(path.join(args.save_dir, 'figure10.png'), dpi=DPI)

    if args.plot:
        plt.show()