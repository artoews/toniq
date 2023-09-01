import matplotlib.pyplot as plt
import numpy as np
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

series_dirs = [root + s for s in series_dirs_bw125]
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

error = metal1.data - pla1.data
denoised_error = analysis.denoise(error)
signal_ref = analysis.get_typical_level(pla1.data)
mask_hyper = analysis.get_mask_hyper(denoised_error, signal_ref)
mask_hypo = analysis.get_mask_hypo(denoised_error, signal_ref)
mask_artifact = analysis.get_mask_artifact(denoised_error, signal_ref)

mask_none = np.zeros_like(mask_implant)
mask_to_artifact = analysis.combine_masks(mask_implant, mask_empty, mask_none, mask_none, mask_artifact)
mask_to_hypo = analysis.combine_masks(mask_implant, mask_empty, mask_none, mask_hypo, mask_artifact)
mask_to_all = analysis.combine_masks(mask_implant, mask_empty, mask_hyper, mask_hypo, mask_artifact)

volumes = (metal1.data, mask_to_artifact, mask_to_hypo, mask_to_all)
titles = ('Metal Image', 'Masks: Implant + Artifact (30%)', '+ Hypointense (-60%)', '+ Hyperintense (+60%)')
fig, tracker = plotVolumes(volumes, 1, len(volumes), titles=titles, figsize=(16, 8))

plt.show()
