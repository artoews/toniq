import argparse
import numpy as np
"""Make Figure 6 for paper.

"""
import matplotlib.pyplot as plt

from os import path, makedirs

from toniq import ia
from toniq.plot import imshow2, remove_ticks, label_encode_dirs, label_slice_pos
from toniq.plot_params import *

fse_arrow_start = (4, 77)
fse_arrow_stop = (19, 77)

def plot_arrow(ax, arrow, color='black'):
    start, stop = arrow
    ax.annotate("", xy=stop, xytext=start, arrowprops=dict(facecolor=color, edgecolor=color, width=2, headwidth=8, headlength=8))

def plot_panel(ax, image, cmap=CMAP['image'], vmin=0, vmax=1.5):
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    remove_ticks(ax)

def plot_row(axes, plastic, metal, ia_map, slc1, slc2, lim=0.8, pad=0, arrow=None):
    _, _, ax1, ax2 = imshow2(axes[0], plastic, slc1, slc2, pad=pad)
    label_slice_pos(ax1, 1, slc2, slc1)
    label_slice_pos(ax2, -1, slc1, slc2)
    label_encode_dirs(ax1)
    label_encode_dirs(ax2, x_label='z') 
    _, _, ax1, ax2 = imshow2(axes[1], metal, slc1, slc2, pad=pad)
    label_slice_pos(ax1, 1, slc2, slc1)
    label_slice_pos(ax2, -1, slc1, slc2)
    if arrow is not None:
        plot_arrow(ax1, arrow)
    im, _, ax1, ax2 = imshow2(axes[2], ia_map, slc1, slc2, vmin=-lim, vmax=lim, cmap=CMAP['artifact'], pad=pad)
    label_slice_pos(ax1, 1, slc2, slc1)
    if arrow is not None:
        plot_arrow(ax1, arrow)
    label_slice_pos(ax2, -1, slc1, slc2)
    ia.colorbar(axes[2], im, lim=lim, offset=0.35)

def print_statistics(ia_map, threshold=0.4):
    ia_mask = np.abs(ia_map) > threshold
    print('IA Mask size (in voxels): {} out of {}'.format(np.sum(ia_mask), ia_mask.size))

p = argparse.ArgumentParser(description='Make figure 6')
p.add_argument('save_dir', type=str, help='path where figure is saved')
p.add_argument('root1', type=str, help='path to main.py output folder 1')
p.add_argument('root2', type=str, help='path to main.py output folder 2')
p.add_argument('-y', '--y_slice', type=int, default=66, help='relative position of z slice (after crop); default=66')
p.add_argument('-z', '--z_slice', type=int, default=18, help='relative position of z slice (after crop); default=18')
p.add_argument('-p', '--plot', action='store_true', help='show plots')

if __name__ == '__main__':

    args = p.parse_args()
    if not path.exists(args.save_dir):
        makedirs(args.save_dir)
    
    slc1 = (slice(None), slice(None), args.z_slice)
    slc2 = (slice(None), args.y_slice, slice(None))

    fig, axes = plt.subplots(nrows=2, ncols=3,
                             figsize=(FIG_WIDTH[2], FIG_WIDTH[2]*0.51),
                             gridspec_kw={'wspace': 0.03, 'hspace': 0.03, 'left': 0.04, 'right': 0.9, 'bottom': 0.05, 'top': 0.92}
                             )
    axes[0, 0].set_title('Plastic')
    axes[0, 1].set_title('Metal')
    axes[0, 2].set_title('Intensity Artifact (IA)')
    axes[0, 0].set_ylabel('2D FSE')
    axes[1, 0].set_ylabel('MAVRIC-SL')

    for i in range(2):
        root = (args.root1, args.root2)[i]
        plastic = np.load(path.join(root, 'ia-plastic.npy'))
        metal = np.load(path.join(root, 'ia-metal.npy'))
        ia_map = np.load(path.join(root, 'ia-map.npy'))
        if i == 0:
            arrow = (fse_arrow_start, fse_arrow_stop)
        else:
            arrow = None
        plot_row(axes[i], plastic, metal, ia_map, slc1, slc2, arrow=arrow)
        # print_statistics(ia_map)

    plt.savefig(path.join(args.save_dir, 'figure6.png'), dpi=DPI, pad_inches=0)
    plt.savefig(path.join(args.save_dir, 'figure6.pdf'), dpi=DPI, pad_inches=0)


    if args.plot:
        plt.show()