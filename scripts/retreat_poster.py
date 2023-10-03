import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from pathlib import Path
from skimage import morphology

import analysis
import dicom

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

def plot(image, dist=None, alpha=None, title=None, cmap='gray', vmin=0, vmax=1):
    nx, ny, nz = image.shape
    fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [ny, nz]}, figsize=(10, 10))
    slc1 = (slice(None), slice(None), 34)
    slc2 = (slice(None), 55, slice(None))
    axes[0].imshow(image[slc1], cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].imshow(image[slc2], cmap=cmap, vmin=vmin, vmax=vmax)
    fig.suptitle(title)
    axes[0].set_title('In-plane')
    axes[1].set_title('Reformat')
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    # axes[0].set_ylabel('x')
    # axes[0].set_xlabel('y')
    # axes[1].set_xlabel('z')
    if dist is not None:
        cmap = 'Spectral'
        mappable = axes[0].imshow(dist[slc1], alpha=alpha[slc1], vmin=-2, vmax=2, cmap=cmap)
        axes[1].imshow(dist[slc2], alpha=alpha[slc2], vmin=-5, vmax=5, cmap=cmap)
        cbar = fig.colorbar(mappable, ax=axes, orientation='horizontal', fraction=.1, ticks=[-2.5, 2.5])
        cbar.ax.set_xticklabels(['', ''])
    return fig, axes

artifact_masks = analysis.combine_masks_2(mask_implant, mask_empty, mask_hyper, mask_hypo, mask_artifact)
deformation_field = np.load('deformation_field_bw125.npy')
distortion = deformation_field[..., 0]

slc = (slice(25, 175), slice(50, 200), slice(10, 70))
pla = pla1.data[slc]
metal = metal1.data[slc]
artifact_masks = artifact_masks[slc]

slc = (slice(10, 130), slice(20, 135), slice(None))

fig, axes = plot(pla[slc], title='Plastic')
fig, axes = plot(metal[slc], title='Metal')
alpha = np.ones_like(artifact_masks)
alpha[artifact_masks!=0] = 0
fig, axes = plot(artifact_masks[slc], title='Analysis', dist=distortion[slc], alpha=alpha[slc], vmin=0.25, vmax=1)

plt.show()
