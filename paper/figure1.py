import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from os import path, makedirs

from plot_params import *
from util import equalize, load_series

exam_root = '/Users/artoews/root/data/mri/240202/14446_dicom'
save_dir = '/Users/artoews/root/code/projects/metal-phantom/feb2'
series_list = ('Series24', 'Series30', 'Series34')
series_names = ('Plastic', 'Metal, 2D FSE', 'Metal, MAVRIC-SL')
slice_names = ('Slice 1', 'Slice 2', 'Reformat')

slc = (slice(38, 158), slice(66, 186), slice(None))
slices = (
    (slc[0], slc[1], 4),
    (slc[0], slc[1], 34),
    (slc[0], 126, slc[2])
)

if not path.exists(save_dir):
    makedirs(save_dir)

def plot_panel(ax, image, cmap=CMAP['image'], vmin=0, vmax=1):
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])

def plot_line(ax, position):
    ax.plot([position, position], [0, slc[0].stop - slc[0].start], color='red', linewidth=1, linestyle=':')

def label_panel(fig, ax, label):
    trans = mtransforms.ScaledTranslation(4e-2, -5e-2, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans, verticalalignment='top', color='white')

images = [load_series(exam_root, series_name).data for series_name in series_list]
images = equalize(np.stack(images)) / 1.5

ratio = images[0][slices[2]].shape[1] / images[0][slices[0]].shape[1]
fig, axes = plt.subplots(nrows=len(series_list), ncols=len(slices), figsize=(FIG_WIDTH[0], FIG_WIDTH[0]*1.195), width_ratios=(1, 1, ratio), gridspec_kw={'wspace': 0, 'hspace': 0})
# labels = [['(A)', '(B)', '(C)'], ['(D)', '(E)', '(F)'], ['(G)', '(H)', '(I)']]

for i in range(len(series_list)):
    for j in range(len(slices)):
        plot_panel(axes[i, j], images[i][slices[j]])
        if j < len(slices) - 1:
            plot_line(axes[i, -1], slices[j][-1])
        axes[0, j].set_title(slice_names[j])
        # label_panel(fig, axes[i, j], labels[i][j])
    axes[i, 0].set_ylabel(series_names[i])

plt.subplots_adjust(wspace=0.025, hspace=0.01)
plt.savefig(path.join(save_dir, 'figure1.png'), dpi=DPI, bbox_inches='tight')
plt.show()
