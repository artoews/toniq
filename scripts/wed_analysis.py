import matplotlib.pyplot as plt
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
    '230830/13511_dicom/Series3',
    '230830/13511_dicom/Series15',
    ''
]
series_dirs_msl = [
    '230830/13511_dicom/Series21',
    '230830/13511_dicom/Series5',
    '230830/13511_dicom/Series17',
    ''
]

series_dirs = [root + s for s in series_dirs_msl]
pla1_files = Path(series_dirs[0]).glob('*MRDC*')
pla2_files = Path(series_dirs[1]).glob('*MRDC*')
metal1_files = Path(series_dirs[2]).glob('*MRDC*')
# metal2_files = Path(series_dirs[3]).glob('*MRDC*')
pla1 = dicom.load_series(pla1_files)
pla2 = dicom.load_series(pla2_files)
metal1 = dicom.load_series(metal1_files)
# metal2 = dicom.load_series(metal2_files)

pla1.data = analysis.normalize(pla1.data)
metal1.data = analysis.equalize(metal1.data, pla1.data)
pla2.data = analysis.equalize(pla2.data, pla1.data)

mask_empty = analysis.get_mask_empty(pla1.data)
mask_implant = analysis.get_mask_implant(mask_empty)

error = metal1.data - pla1.data
denoised_error = analysis.denoise(error)
signal_ref = analysis.get_typical_level(pla1.data)

mask_hyper = analysis.get_mask_hyper(denoised_error, signal_ref)
mask_hypo = analysis.get_mask_hypo(denoised_error, signal_ref)
mask_artifact = analysis.get_mask_artifact(denoised_error, signal_ref)

mega_mask = analysis.combine_masks(mask_implant, mask_empty, mask_hyper, mask_hypo, mask_artifact)

# plotting
def plot_panel(volumes, titles):
    fig, tracker = plotVolumes(volumes, 1, len(volumes), titles=titles, figsize=(15, 8))
    return fig, tracker

fig1, tracker1 = plot_panel((pla1.data, metal1.data, (error+1)/2, (denoised_error+1)/2),
                            ('Plastic', 'Metal', 'Error = Metal - Plastic', 'Denoised Error'))
fig2, tracker2 = plot_panel((metal1.data, signal_ref, (denoised_error+1)/2, mega_mask),
                            ('Metal', 'Reference level (from Plastic)', 'Denoised Error', 'Composite Mask'))
repeat_error = pla1.data - pla2.data
fig3, tracker3 = plot_panel((pla2.data, pla1.data, (repeat_error+1)/2),
                            ('At session start', 'At session end', 'Error'))

plt.show()