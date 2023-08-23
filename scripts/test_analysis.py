# %%
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import analysis
import dicom
from plot import plotVolumes

# %%
# two scan series with FSE @ 15.63 kHz RBW, 1.2mm iso
series_dirs = [
    '/bmrNAS/people/artoews/data/scans/230801/13295_dicom/Series6',
    '/bmrNAS/people/artoews/data/scans/230801/13295_dicom/Series17',
]
plastic_files = Path(series_dirs[0]).glob('*MRDC*')
metal_files = Path(series_dirs[1]).glob('*MRDC*')
plastic = dicom.load_series(plastic_files)
metal = dicom.load_series(metal_files)
margin = 0.3
fs = 4

# %%
hypo_mask = analysis.mask_signal_void(metal.data, reference_image=plastic.data, threshold=1-margin, filter_size=fs)

# %%
hyper_mask = analysis.mask_signal_void(-metal.data, reference_image=-plastic.data, threshold=1+margin, filter_size=fs)

# %%
artifact_energy = analysis.energy(metal.data, plastic.data, kernel='median', size=fs)

# %%
plastic_data = np.clip(plastic.data, 0, 1500)
metal_data = np.clip(metal.data, 0, 1500)
diff = np.clip(np.abs(metal.data - plastic.data), 0, 500)
artifact_energy = np.clip(np.log(artifact_energy), 0, 10)
volumes = (plastic_data, metal_data, diff, artifact_energy, hypo_mask, hyper_mask)
plotVolumes(volumes, 1, 6, vmax=None, titles=('Plastic', 'Metal', 'diff', 'artifact energy', 'hypo mask', 'hyper mask'), figsize=(18, 6))


