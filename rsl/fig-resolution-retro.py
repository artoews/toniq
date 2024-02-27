import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp

from os import path
from plot_params import *
from slice_params import *
from util import normalize, equalize, load_series_from_path

series_path = '/Users/artoews/root/data/mri/240202/14446_dicom/Series4'
save_dir = '/Users/artoews/root/code/projects/metal-phantom/rsl'
load = False

slc1 = (slice(None), slice(None), 30)
slc2 = (slice(38*2, 158*2), slice(66*2, 186*2))

if not load:
    shapes = ((512, 512), (256, 256), (256, 172), (172, 256), (172, 172))
    image_ref = load_series_from_path(series_path).data[slc1]
    image_ref = normalize(image_ref)
    images = [None] * len(shapes)
    kspaces = [None] * len(shapes)
    kspace_ref = sp.fft(image_ref)
    for i in range(len(shapes)):
        k = sp.resize(sp.resize(kspace_ref, shapes[i]), image_ref.shape)
        images[i] = np.abs(sp.ifft(k))
        kspaces[i] = np.abs(k)
    images = equalize(images)
    np.save(path.join(save_dir, 'retro-images.npy'), images)
    np.save(path.join(save_dir, 'retro-kspaces.npy'), kspaces)

else:
    np.load(path.join(save_dir, 'retro-images.npy'))
    np.load(path.join(save_dir, 'retro-kspaces.npy'))

fig, axes = plt.subplots(figsize=(10, 4), nrows=2, ncols=len(shapes), layout='constrained')
for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
for i in range(len(shapes)):
    axes[0, i].imshow(images[i][slc2], vmin=0, vmax=1, cmap=CMAP['image'])
    axes[1, i].imshow(np.log(kspaces[i]+1), vmin=0, vmax=0.5, cmap=CMAP['image'])
# axes[0, 0].set_ylabel('Image')
# axes[1, 0].set_ylabel('K-Space')
plt.savefig(path.join(save_dir, 'resolution-retro-images.png'), dpi=300, transparent=True)
    
fig, axes = plt.subplots(figsize=(3, 5), nrows=3, ncols=1, layout='constrained')
x = np.arange(-256, 256)
sizes = (256, 172)
styles = ('solid', 'dashed')
for i in range(len(sizes)):
    mask = sp.resize(sp.resize(np.ones(len(x)), (sizes[i],)), (len(x),))
    axes[0].plot(x, mask, linestyle=styles[i])
    psf = np.real(sp.ifft(mask))
    axes[1].plot(x, psf / np.max(psf), linestyle=styles[i])
    axes[2].plot(x, psf / np.max(psf), linestyle=styles[i])
axes[0].set_title('K-Space Mask')
axes[0].set_xticks([-256, -128, 0, 128, 256])
axes[0].set_xticks(np.arange(-256, 257, 128))
axes[0].set_yticks([0, 0.5, 1])
axes[1].set_title('PSF (normalized)')
axes[1].set_xlim([-10, 10])
axes[1].set_yticks([0, 0.5, 1])
axes[2].set_title('PSF Inset')
axes[2].set_xlim([-2, 2])
axes[2].set_yticks([0, 0.5, 1])
axes[2].grid()
plt.savefig(path.join(save_dir, 'resolution-retro-psf.png'), dpi=300, transparent=True)

plt.show()
