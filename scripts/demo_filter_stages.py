import matplotlib.pyplot as plt
from pathlib import Path

import analysis
import dicom
from plot import plotVolumes

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

series_dirs = [root + s for s in series_dirs_msl]
pla1_files = Path(series_dirs[0]).glob('*MRDC*')
pla2_files = Path(series_dirs[1]).glob('*MRDC*')
metal1_files = Path(series_dirs[2]).glob('*MRDC*')
pla1 = dicom.load_series(pla1_files)
pla2 = dicom.load_series(pla2_files)
metal1 = dicom.load_series(metal1_files)

metal1.data = analysis.equalize(metal1.data, pla1.data)
pla2.data = analysis.equalize(pla2.data, pla1.data)

mask_empty = analysis.get_mask_empty(pla1.data)
mask_implant = analysis.get_mask_implant(mask_empty)

error = metal1.data - pla1.data
denoised_error = analysis.denoise(error)
is_denoised = True
signal_ref = analysis.get_typical_level(pla1.data)
stages_hyper = analysis.get_mask_extrema(denoised_error, signal_ref, 0.6, is_denoised, return_stages=True)
stages_hypo = analysis.get_mask_extrema(denoised_error, signal_ref, -0.6, is_denoised, return_stages=True)
stages_artifact = analysis.get_mask_extrema(denoised_error, signal_ref, 0.3, is_denoised, mag=True, return_stages=True)

def plot_stages(stages, threshold, vmax=3e3):
    clean_mask, raw_mask, filtered_error = stages
    if threshold == 0.3:
        filter_name = 'max-of-mag filter'
    elif threshold > 0:
        filter_name = 'max filter'
    else:
        filter_name = 'min filter'
    threshold_name = 'threshold outside {}%'.format(threshold * 100)
    volumes = ((error + vmax) / (2 * vmax),
               (denoised_error + vmax) / (2 * vmax),
               (filtered_error + vmax) / (2 * vmax),
               raw_mask,
               clean_mask)
    titles = ('metal-plastic difference', 'median filter', filter_name, threshold_name, 'opening filter')
    fig, tracker = plotVolumes(volumes, 1, len(volumes), titles=titles, figsize=(16, 8))
    return fig, tracker

fig1, tracker1 = plot_stages(stages_hyper, 0.6)
fig2, tracker2 = plot_stages(stages_hypo, -0.6)
fig3, tracker3 = plot_stages(stages_artifact, 0.3)

plt.show()
