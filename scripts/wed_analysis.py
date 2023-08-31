import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from skimage import morphology
import scipy.ndimage as ndi

import analysis
import dicom
from plot import plotVolumes

root = '/Users/artoews/root/data/mri/'
# root = 'bmrNAS/people/artoews/data/scans/'

series_dirs_bw125 = [
    '230817/13427_dicom/Series3',
    '230817/13427_dicom/Series7',
    '230817/13427_dicom/Series11',
    '230817/13427_dicom/Series15'
]
series_dirs_msl = [
    '230817/13427_dicom/Series6',
    '230817/13427_dicom/Series10',
    '230817/13427_dicom/Series14',
    '230817/13427_dicom/Series18'
]
series_dirs_bw31 = [
    '230817/13427_dicom/Series4',
    '230817/13427_dicom/Series8',
    '230817/13427_dicom/Series12',
    '230817/13427_dicom/Series16'
]
series_dirs_bw125 = [
    '230830/13511_dicom/Series19',
    '',
    '230830/13511_dicom/Series15',
    ''
]
series_dirs_msl = [
    '230830/13511_dicom/Series21',
    '',
    '230830/13511_dicom/Series17',
    ''
]

series_dirs = [root + s for s in series_dirs_msl]
pla1_files = Path(series_dirs[0]).glob('*MRDC*')
# pla2_files = Path(series_dirs[1]).glob('*MRDC*')
metal1_files = Path(series_dirs[2]).glob('*MRDC*')
# metal2_files = Path(series_dirs[3]).glob('*MRDC*')
pla1 = dicom.load_series(pla1_files)
# pla2 = dicom.load_series(pla2_files)
metal1 = dicom.load_series(metal1_files)
# metal2 = dicom.load_series(metal2_files)

metal1.data = metal1.data / np.median(metal1.data) * np.median(pla1.data)  # equalize images

# avg_signal_pla = np.mean(pla1.data[180:200, 180:200, 35])
# avg_signal_metal = np.mean(metal1.data[180:200, 180:200, 35])

mask_empty = analysis.get_mask_empty(pla1.data)
mask_implant = analysis.get_mask_implant(mask_empty)

error = metal1.data - pla1.data
denoised_error = analysis.denoise(error)
signal_ref = analysis.get_typical_level(pla1.data)

mask_hyper = analysis.get_mask_hyper(denoised_error, signal_ref)
mask_hypo = analysis.get_mask_hypo(denoised_error, signal_ref)
mask_artifact = analysis.get_mask_artifact(denoised_error, signal_ref)

mega_mask = np.zeros(mask_empty.shape)
mega_mask[mask_artifact] = 3
mega_mask[mask_hyper] = 4
mega_mask[mask_hypo] = 2

mega_mask[mask_empty] = 0
mega_mask[mask_implant] = 1

# plotting
def plot_panel(volumes, titles):
    nvols = len(volumes)
    # vmin = [0, -5e3, 0]
    # vmax = [1e4, 5e3, 1]
    vmin = [None] * len(volumes)
    vmax = [None] * len(volumes)
    cmaps = ['gray'] * len(volumes)
    fig, tracker = plotVolumes(volumes, 1, len(volumes), vmin, vmax, cmaps, titles=titles, figsize=(15, 8))
    return fig, tracker

# fig1, tracker1 = plot_panel((pla1.data, np.abs(metal1.data - pla1.data), mask_unexcited, mask_artifact),
#                             ('Plastic', 'Abs Diff', 'Unexcited Mask', 'Artifact')
#                             )
error = metal1.data - pla1.data
fig2, tracker2 = plot_panel((metal1.data, error, mega_mask, signal_ref),
                            ('Metal', 'Error', 'Mega Mask', 'Signal Ref'))
# fig2, tracker2 = plot_panel(metal2.data, metal2.data - pla2.data, intensity_artifact, 'Metal')

# fig1, tracker1 = plot_panel(pla1.data, pla2.data, noise_pla, noise_pla_filtered, 'PLA 1', 'PLA 2')
# fig2, tracker2 = plot_panel(metal1.data, metal2.data, noise_metal, noise_metal_filtered, 'Metal 1', 'Metal 2')
# fig3, tracker3 = plot_panel(pla1.data, metal1.data, artifact, artifact_filtered, 'PLA 1', 'Metal 1')

plt.show()