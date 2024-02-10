import numpy as np
import matplotlib.pyplot as plt
from os import path, makedirs

from plot_params import *
from util import equalize, load_series

exam_root = '/Users/artoews/root/data/mri/240202/14446_dicom'
save_dir = '/Users/artoews/root/code/projects/metal-phantom/feb2'
series_list = ['Series30', 'Series34', 'Series24']

slc = (slice(38, 158), slice(66, 186), slice(None))
slices = (
    (slc[0], slc[1], 34),
    (slc[0], slc[1], 5),
    (slc[0], 126, slc[2])
)

if not path.exists(save_dir):
    makedirs(save_dir)

def plot_panel(ax, image, cmap=CMAP['image'], vmin=0, vmax=1):
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')

def plot_line(ax, position):
    ax.plot([position, position], [0, slc[0].stop - slc[0].start], color='red', linewidth=1, linestyle=':')

images = [load_series(exam_root, series_name).data for series_name in series_list]
images = equalize(np.stack(images)) / 1.5

ratio = images[0][slices[2]].shape[1] / images[0][slices[0]].shape[1]
fig, axes = plt.subplots(nrows=len(series_list), ncols=len(slices), figsize=(5, 6), width_ratios=(1, 1, ratio))

for i in range(len(series_list)):
    for j in range(len(slices)):
        plot_panel(axes[i, j], images[i][slices[j]])
        if j < len(slices) - 1:
            plot_line(axes[i, -1], slices[j][-1])

plt.subplots_adjust(wspace=0.025, hspace=0.01)
plt.savefig(path.join(save_dir, 'figure1.png'), dpi=300, bbox_inches='tight', pad_inches=0)
