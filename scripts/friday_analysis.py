import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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
series_dirs_bw31 = [
    '230817/13427_dicom/Series4',
    '230817/13427_dicom/Series8',
    '230817/13427_dicom/Series12',
    '230817/13427_dicom/Series16'
]

series_dirs = [root + s for s in series_dirs_bw125]
pla1_files = Path(series_dirs[0]).glob('*MRDC*')
pla2_files = Path(series_dirs[1]).glob('*MRDC*')
metal1_files = Path(series_dirs[2]).glob('*MRDC*')
metal2_files = Path(series_dirs[3]).glob('*MRDC*')
pla1 = dicom.load_series(pla1_files)
pla2 = dicom.load_series(pla2_files)
metal1 = dicom.load_series(metal1_files)
metal2 = dicom.load_series(metal2_files)

# parameters
fs = 2
kernel = 'gaussian'
energy = False

# noise energy
noise_pla_filtered, noise_pla = analysis.error(pla2.data, pla1.data, kernel=kernel, size=fs, energy=energy)
noise_metal_filtered, noise_metal = analysis.error(metal2.data, metal1.data, kernel=kernel, size=fs, energy=energy)

# artifact energy
artifact_filtered, artifact = analysis.error(metal1.data, pla1.data, kernel=kernel, size=fs, energy=energy)

# plotting
def plot_panel(im1, im2, diff, filtered_diff, im1_title, im2_title):
    volumes = (im1, im2, diff, filtered_diff)
    titles = (im1_title, im2_title, 'diff', 'filtered diff')
    nvols = len(volumes)
    vmin = [0, 0, -5e3, -5e3]
    vmax = [1e4, 1e4, 5e3, 5e3]
    cmaps = ['gray' for _ in range(nvols)]
    fig, tracker = plotVolumes(volumes, 1, len(volumes), vmin, vmax, cmaps, titles=titles, figsize=(18, 6))
    return fig, tracker

fig1, tracker1 = plot_panel(pla1.data, pla2.data, noise_pla, noise_pla_filtered, 'PLA 1', 'PLA 2')
fig2, tracker2 = plot_panel(metal1.data, metal2.data, noise_metal, noise_metal_filtered, 'Metal 1', 'Metal 2')
fig3, tracker3 = plot_panel(pla1.data, metal1.data, artifact, artifact_filtered, 'PLA 1', 'Metal 1')

plt.show()