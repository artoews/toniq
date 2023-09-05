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

series_dirs = [root + s for s in series_dirs_bw31]
image1_files = Path(series_dirs[0]).glob('*MRDC*')
image2_files = Path(series_dirs[1]).glob('*MRDC*')
image1 = dicom.load_series(image1_files)
image2 = dicom.load_series(image2_files)

image1.data = analysis.normalize(image1.data)
image2.data = analysis.equalize(image2.data, image1.data)

mask_empty = analysis.get_mask_empty(image1.data)
mask_implant = analysis.get_mask_implant(mask_empty)
mask_signal = analysis.get_mask_signal(image1.data, image2.data)

noise = image2.data - image1.data
snr, signal, noise_std, mask_signal = analysis.signal_to_noise(image1.data, image2.data, mask_signal, mask_empty)

factor = 20
noise = (factor * noise + 1) / 2
noise_std = (factor * noise_std + 1) / 2
volumes = (image1.data, image2.data, noise, noise_std, signal, mask_signal, snr / 40)
titles = ('Image 1', 'Image 2', 'Noise', 'Noise St. Dev.', 'Signal', 'Mask Signal', 'SNR')
fig, tracker = plotVolumes(volumes, 1, len(volumes), titles=titles, figsize=(16, 8))

plt.show()