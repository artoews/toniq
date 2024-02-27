import matplotlib.pyplot as plt
import numpy as np

from os import path
from plot import imshow2
from plot_params import *
from slice_params import *
from util import equalize, load_series_from_path

series_paths = [
    '/Users/artoews/root/data/mri/240202/14446_dicom/Series24', # empty-plastic, 2D-FSE
    '/Users/artoews/root/data/mri/240202/14446_dicom/Series30', # empty-metal, 2D-FSE
    '/Users/artoews/root/data/mri/240202/14446_dicom/Series34', # empty-metal, MAVRIC-SL
]
# series_paths = [
#     '/Users/artoews/root/data/mri/240202/14446_dicom/Series14', # lattice-plastic, 2D-FSE
#     '/Users/artoews/root/data/mri/240202/14446_dicom/Series18', # lattice-metal, 2D-FSE
#     '/Users/artoews/root/data/mri/240202/14446_dicom/Series20', # lattice-metal, MAVRIC-SL
# ]

save_dir = '/Users/artoews/root/code/projects/metal-phantom/rsl'
field_strength = 3  # T
gyromagnetic_ratio = 42.58 # MHz/T

slc1 = (slice(38, 158), slice(66, 186), 5)
slc2 = (slice(38, 158), slice(66, 186), 34)
slc3 = (slice(38, 158), 126, slice(None))

images = np.stack([load_series_from_path(p).data for p in series_paths])
images = equalize(images)

fig, axes = plt.subplots(figsize=(10, 4), nrows=2, ncols=3, layout='constrained')
imshow2(axes[0, 0], images[0], slc1, slc3, y_label='readout', x1_label='phase', x2_label='slice')
imshow2(axes[0, 1], images[1], slc1, slc3)
imshow2(axes[0, 2], images[2], slc1, slc3)
imshow2(axes[1, 0], images[0], slc2, slc3, y_label='readout', x1_label='phase', x2_label='slice')
imshow2(axes[1, 1], images[1], slc2, slc3)
imshow2(axes[1, 2], images[2], slc2, slc3)

plt.savefig(path.join(save_dir, 'metal-intro.png'), dpi=300)
plt.show()

