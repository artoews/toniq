import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import analysis
import dicom
from plot import plotVolumes
from util import safe_divide

root = '/Users/artoews/root/data/mri/'

series_dirs_bw125 = [
    '230830/13511_dicom/Series19',
    '230830/13511_dicom/Series3',
    '230830/13511_dicom/Series15',
]
series_dirs_msl = [
    '230830/13511_dicom/Series21',
    '230830/13511_dicom/Series5',
    '230830/13511_dicom/Series17',
]

series_dirs_bw125 = [
    '231021/13882_dicom/Series3',
    '231021/13882_dicom/Series7',
    '231021/13882_dicom/Series30',
]
 
series_dirs_msl = [
    '231021/13882_dicom/Series21',
    '231021/13882_dicom/Series25',
    '231021/13882_dicom/Series48',
]

series_dirs = [root + s for s in series_dirs_msl]
pla1_files = Path(series_dirs[0]).glob('*MRDC*')
pla2_files = Path(series_dirs[1]).glob('*MRDC*')
metal1_files = Path(series_dirs[2]).glob('*MRDC*')
pla1 = dicom.load_series(pla1_files)
pla2 = dicom.load_series(pla2_files)
metal1 = dicom.load_series(metal1_files)

pla1.data = analysis.normalize(pla1.data)
metal1.data = analysis.equalize(metal1.data, pla1.data)
pla2.data = analysis.equalize(pla2.data, pla1.data)

mask_empty = analysis.get_mask_empty(pla1.data)
mask_implant = analysis.get_mask_implant(mask_empty)
mask_signal = analysis.get_mask_signal(pla1.data)

error = metal1.data - pla1.data
signal_ref = analysis.get_typical_level(pla1.data, mask_signal, mask_implant)
stages_artifact = analysis.get_mask_extrema(error, signal_ref, 0.3, 'mean', abs_margin=True)
stages_artifact_hyper = analysis.get_mask_extrema(error, signal_ref, 0.3, 'mean', abs_margin=False)
stages_artifact_hypo = analysis.get_mask_extrema(error, signal_ref, -0.3, 'mean', abs_margin=False)

def plot_stages(stages, threshold, vmax=1):
    mask, filtered_error = stages
    threshold_name = 'threshold at {}%'.format(threshold * 100)
    volumes = (pla1.data / vmax,
               metal1.data / vmax,
               safe_divide(error, signal_ref) + 0.5,
               safe_divide(filtered_error, signal_ref) + 0.5,
               # (error + vmax) / (2 * vmax),
               # (filtered_error + vmax) / (2 * vmax),
               signal_ref / vmax,
               mask)
    titles = ('plastic', 'metal', 'difference', 'filter of diff', 'signal level', threshold_name)
    fig, tracker = plotVolumes(volumes, 1, len(volumes), titles=titles, figsize=(16, 8))
    return fig, tracker

fig1, tracker1 = plot_stages(stages_artifact, 0.3)
fig2, tracker2 = plot_stages(stages_artifact_hyper, 0.3)
fig3, tracker3 = plot_stages(stages_artifact_hypo, -0.3)

plt.show()
