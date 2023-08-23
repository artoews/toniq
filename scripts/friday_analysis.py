import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import analysis
import dicom
from plot import plotVolumes


series_dirs_bw125 = [
    '/bmrNAS/people/artoews/data/scans/230817/13427_dicom/Series3',
    '/bmrNAS/people/artoews/data/scans/230817/13427_dicom/Series7',
    '/bmrNAS/people/artoews/data/scans/230817/13427_dicom/Series11',
    '/bmrNAS/people/artoews/data/scans/230817/13427_dicom/Series15'
]
series_dirs_bw31 = [
    '/bmrNAS/people/artoews/data/scans/230817/13427_dicom/Series4',
    '/bmrNAS/people/artoews/data/scans/230817/13427_dicom/Series8',
    '/bmrNAS/people/artoews/data/scans/230817/13427_dicom/Series12',
    '/bmrNAS/people/artoews/data/scans/230817/13427_dicom/Series16'
]

series_dirs = series_dirs_bw125
pla1_files = Path(series_dirs[0]).glob('*MRDC*')
pla2_files = Path(series_dirs[1]).glob('*MRDC*')
metal1_files = Path(series_dirs[2]).glob('*MRDC*')
metal2_files = Path(series_dirs[3]).glob('*MRDC*')
pla1 = dicom.load_series(pla1_files)
pla2 = dicom.load_series(pla2_files)
metal1 = dicom.load_series(metal1_files)
metal2 = dicom.load_series(metal2_files)

# parameters
fs = 5
kernel = 'max'

# noise energy
noise_energy_pla = analysis.energy(pla2.data, pla1.data, kernel=kernel, size=fs)
noise_energy_metal = analysis.energy(metal2.data, metal1.data, kernel=kernel, size=fs)

# artifact energy
artifact_energy = analysis.energy(metal1.data, pla1.data, kernel=kernel, size=fs)

# TODO try filtering the signed diff with a small kernel. See notes from last BH meeting.

# plotting
def plot_panel(im1, im2, filtered_energy, im1_title, im2_title):
    diff = im2 - im1
    energy = analysis.energy(im2, im1)
    volumes = (im1, im2, diff, energy, filtered_energy)
    titles = (im1_title, im2_title, 'diff', 'energy of diff', 'filtered energy of diff')
    nvols = len(volumes)
    vmin = [0, 0, -5e3, 0, 0]
    vmax = [1e4, 1e4, 5e3, 1e6, 1e6]
    cmaps = ['gray' for _ in range(nvols)]
    fig, tracker = plotVolumes(volumes, 1, len(volumes), vmin, vmax, cmaps, titles=titles, figsize=(18, 6))
    return fig, tracker

fig1, tracker1 = plot_panel(pla1.data, pla2.data, noise_energy_pla, 'PLA 1', 'PLA 2')
fig2, tracker2 = plot_panel(metal1.data, metal2.data, noise_energy_metal, 'Metal 1', 'Metal 2')
fig3, tracker3 = plot_panel(pla1.data, metal1.data, artifact_energy, 'PLA 1', 'Metal 1')

plt.show()