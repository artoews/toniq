import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from skimage import morphology

import analysis
import dicom
from plot import plotVolumes

root = '/Users/artoews/root/data/mri/'

series_dirs_bw31 = [
    '230830/13511_dicom/Series4',
    '230830/13511_dicom/Series8',
    '230830/13511_dicom/Series12'
]
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

mask_implant, mask_empty, mask_hyper, mask_hypo, mask_artifact = analysis.get_all_masks(pla1.data, metal1.data)
mask_artifact = morphology.dilation(mask_artifact, morphology.ball(2))

mask_none = np.zeros_like(mask_implant)
mask_to_artifact = analysis.combine_masks(mask_implant, mask_empty, mask_none, mask_none, mask_artifact)
mask_to_hypo = analysis.combine_masks(mask_implant, mask_empty, mask_none, mask_hypo, mask_artifact)
mask_to_all = analysis.combine_masks(mask_implant, mask_empty, mask_hyper, mask_hypo, mask_artifact)

clean_metal = metal1.data.copy()
clean_metal[mask_to_all != 2/5] = 0

volumes = (pla1.data, metal1.data, mask_to_artifact, mask_to_hypo, mask_to_all, clean_metal)
titles = ('Plastic', 'Metal', 'Masks: Implant + Artifact (30%)', '+ Hypointense (-60%)', '+ Hyperintense (+60%)', 'Clean signal')
fig, tracker = plotVolumes(volumes, 1, len(volumes), titles=titles, figsize=(16, 8))

plt.show()
