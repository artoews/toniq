import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import analysis
import dicom
from plot import plotVolumes

root = '/Users/artoews/root/data/mri/'

series_dirs_bw125 = [
    '230830/13511_dicom/Series3',
    '230830/13511_dicom/Series7',
]
series_dirs_bw31 = [
    '230830/13511_dicom/Series4',
    '230830/13511_dicom/Series8',
]
series_dirs_msl = [
    '230830/13511_dicom/Series5',
    '230830/13511_dicom/Series9',
]

series_dirs = [root + s for s in series_dirs_msl]
image1_files = Path(series_dirs[0]).glob('*MRDC*')
image2_files = Path(series_dirs[1]).glob('*MRDC*')
image1 = dicom.load_series(image1_files)
image2 = dicom.load_series(image2_files)

image1.data = analysis.normalize(image1.data)
image2.data = analysis.equalize(image2.data, image1.data)

mask_empty = analysis.get_mask_empty(image1.data)
mask_implant = analysis.get_mask_implant(mask_empty)
mask_signal = analysis.get_mask_signal(image1.data)
exp_signal = analysis.get_typical_level(image1.data, mask_signal, mask_implant)
signal_masked = image1.data.copy()
signal_masked[~mask_signal] = 0

image_diff = 5 * (image2.data - image1.data) + 0.5
image_sum = 0.5 * (image2.data + image1.data)
volumes = (image1.data, image2.data, image_diff, image_sum, mask_signal, signal_masked, exp_signal)
titles = ('Image 1', 'Image 2', 'Difference (5x)', 'Sum (0.5x)', 'Signal Mask', 'Signal Masked', 'Expected Signal')
fig2, tracker2 = plotVolumes(volumes, 1, len(volumes), titles=titles, figsize=(16, 8))

plt.show()