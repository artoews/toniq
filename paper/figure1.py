import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import yaml

from os import path, makedirs
from pathlib import Path

from config import parse_slice  
from plot_params import *
from util import equalize, load_series_from_path

def plot_panel(ax, image, cmap=CMAP['image'], vmin=0, vmax=1):
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])

def plot_line(ax, position):
    ax.plot([position, position], [0, slc[0].stop - slc[0].start], color='red', linewidth=1, linestyle=':')

def label_panel(fig, ax, label):
    trans = mtransforms.ScaledTranslation(4e-2, -5e-2, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans, verticalalignment='top', color='white')

p = argparse.ArgumentParser(description='Make figure 1')
p.add_argument('save_dir', type=str, help='path where figure is saved')
p.add_argument('-c1', '--config1', type=str, default='config/feb2-fse125.yml', help='yaml config file for FSE sequence')
p.add_argument('-c2', '--config2', type=str, default='config/feb2-msl125.yml', help='yaml config file for MAVRIC-SL sequence')
p.add_argument('-s1', '--slice1', type=int, default=4, help='z index of first slice')
p.add_argument('-s2', '--slice2', type=int, default=34, help='z index of second slice')
p.add_argument('-s3', '--slice3', type=int, default=126, help='y index of third slice')
p.add_argument('-p', '--plot', action='store_true', help='show plots')

if __name__ == '__main__':

    args = p.parse_args()
    if not path.exists(args.save_dir):
        makedirs(args.save_dir)

    with open(args.config1, 'r') as file:
        config1 = yaml.safe_load(file)
    with open(args.config2, 'r') as file:
        config2 = yaml.safe_load(file)
    slc = parse_slice(config2)[:2] + (slice(None),)

    slices = (
        slc[:2] + (args.slice1,),
        slc[:2] + (args.slice2,),
        slc[:1] + (args.slice3,) + slc[2:]
    )

    paths = (
        config1['dicom-series']['uniform-plastic'],
        config1['dicom-series']['uniform-metal'],
        config2['dicom-series']['uniform-metal'],
    )
    series_names = ('Plastic', 'Metal, 2D FSE', 'Metal, MAVRIC-SL')
    slice_names = ('Slice 1', 'Slice 2', 'Reformat')

    images = [load_series_from_path(path).data for path in paths]
    images = equalize(np.stack(images))

    ratio = images[0][slices[2]].shape[1] / images[0][slices[0]].shape[1]
    fig, axes = plt.subplots(
        nrows=len(paths), ncols=len(slices),
        figsize=(FIG_WIDTH[0], FIG_WIDTH[0]*1.195),
        width_ratios=(1, 1, ratio),
        gridspec_kw={'wspace': 0, 'hspace': 0}
        )
    
    for i in range(len(paths)):
        for j in range(len(slices)):
            plot_panel(axes[i, j], images[i][slices[j]], vmax=1.5)
            if j < len(slices) - 1:
                plot_line(axes[i, -1], slices[j][-1])
            axes[0, j].set_title(slice_names[j])
            # label_panel(fig, axes[i, j], labels[i][j])
        axes[i, 0].set_ylabel(series_names[i])

    plt.subplots_adjust(wspace=0.025, hspace=0.01)
    plt.savefig(path.join(args.save_dir, 'figure1.png'), dpi=DPI, bbox_inches='tight')

    if args.plot:
        plt.show()
