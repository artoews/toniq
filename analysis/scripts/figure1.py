"""Make Figure 1 for paper.

"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from os import path, makedirs

from toniq.config import read_config, parse_slice, load_volume
from toniq.plot_params import *
from toniq.plot import label_encode_dirs, label_slice_pos
from toniq.util import equalize

def plot_panel(ax, image, cmap=CMAP['image'], vmin=0, vmax=1.5):
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])

def label_panel(fig, ax, label):
    trans = mtransforms.ScaledTranslation(4e-2, -5e-2, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans, verticalalignment='top', color='white')

p = argparse.ArgumentParser(description='Make figure 1')
p.add_argument('save_dir', type=str, help='path where figure is saved')
p.add_argument('-c1', '--config1', type=str, default='fse125.yml', help='data config file for FSE sequence')
p.add_argument('-c2', '--config2', type=str, default='msl125.yml', help='data config file for MAVRIC-SL sequence')
p.add_argument('-s1', '--slice1', type=int, default=4, help='z index of first slice')
p.add_argument('-s2', '--slice2', type=int, default=34, help='z index of second slice')
p.add_argument('-s3', '--slice3', type=int, default=126, help='y index of third slice')
p.add_argument('-p', '--plot', action='store_true', help='show plots')

if __name__ == '__main__':

    args = p.parse_args()
    if not path.exists(args.save_dir):
        makedirs(args.save_dir)

    config1 = read_config(args.config1)
    config2 = read_config(args.config2)
    slc = parse_slice(config2)[:2] + (slice(None),)

    slice_names = ('Slice 1', 'Slice 2', 'Reformat')
    slices = (
        slc[:2] + (args.slice1,),
        slc[:2] + (args.slice2,),
        slc[:1] + (args.slice3,) + slc[2:]
    )

    series_names = ('2D FSE\nPlastic', '2D FSE\nMetal', 'MAVRIC-SL\nMetal')
    images = [
        load_volume(config1, 'uniform-plastic').data,
        load_volume(config1, 'uniform-metal').data,
        load_volume(config2, 'uniform-metal').data
    ]
    images = equalize(np.stack(images))

    ratio = images[0][slices[2]].shape[1] / images[0][slices[0]].shape[1]
    fig, axes = plt.subplots(
        nrows=3, ncols=len(slices),
        figsize=(FIG_WIDTH[0], FIG_WIDTH[0]*1.195),
        width_ratios=(1, 1, ratio),
        gridspec_kw={'wspace': 0, 'hspace': 0}
        )
    
    for i in range(3):
        for j in range(len(slices)):
            plot_panel(axes[i, j], images[i][slices[j]])
            if j < len(slices) - 1:
                label_slice_pos(axes[i, -1], -1, slices[j], slc, label=j+1)
                label_slice_pos(axes[i, j], 1, slices[-1], slc)
            axes[0, j].set_title(slice_names[j])
            # label_panel(fig, axes[i, j], labels[i][j])
        axes[i, 0].set_ylabel(series_names[i])
    
    label_encode_dirs(axes[0, 0])
    label_encode_dirs(axes[0, 1])
    label_encode_dirs(axes[0, 2], x_label='z')

    plt.subplots_adjust(wspace=0.025, hspace=0.01)
    plt.savefig(path.join(args.save_dir, 'figure1.png'), dpi=DPI, bbox_inches='tight')
    plt.savefig(path.join(args.save_dir, 'figure1.pdf'), dpi=DPI, bbox_inches='tight')

    if args.plot:
        plt.show()
