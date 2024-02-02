import numpy as np
import matplotlib.pyplot as plt
from os import path, makedirs

from plot_params import *
from plot import overlay_mask


# root = '/Users/artoews/root/code/projects/metal-phantom/jan26/'
# root = '/Users/artoews/root/code/projects/metal-phantom/jan26-msl/'
root = '/Users/artoews/root/code/projects/metal-phantom/jan21/'
# root = '/Users/artoews/root/code/projects/metal-phantom/jan21-msl/'
slc = (slice(None), slice(None), 19)
res_slc = (slice(None), slice(None), 9)

save_dir = path.join(root, 'figure2')
if not path.exists(save_dir):
    makedirs(save_dir)

lattice_images = np.load(path.join(root, 'distortion', 'images.npy'))
empty_images = np.load(path.join(root, 'artifact', 'images.npy'))
res_images = np.load(path.join(root, 'resolution', 'images.npy'))

implant_mask = np.load(path.join(root, 'artifact', 'implant-mask.npy'))

ia_maps = np.load(path.join(root, 'artifact', 'artifact-maps.npy'))
# snr_maps = np.load(path.join(root, 'snr', 'snr-maps.npy'))
# snr_masks = np.load(path.join(root, 'snr', 'snr-masks.npy'))
gd_maps = np.load(path.join(root, 'distortion', 'gd-maps.npy'))
gd_masks = np.load(path.join(root, 'distortion', 'gd-masks.npy'))
res_maps = np.load(path.join(root, 'resolution', 'res-maps.npy'))
res_masks = np.load(path.join(root, 'resolution', 'res-masks.npy'))

def save_thumbnail(image, title, figsize=(3, 3), cmap=CMAP['image'], vmin=0, vmax=1, mask=None):
    fig, ax = plt.subplots(figsize=figsize)
    plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    if mask is not None:
        overlay_mask(ax, ~mask, color=[0, 0, 0])
    plt.axis('off')
    plt.savefig(path.join(save_dir, title), dpi=300, bbox_inches='tight', pad_inches=0)

save_thumbnail(empty_images[0][slc], 'empty-plastic.png')
save_thumbnail(empty_images[1][slc], 'empty-metal.png')
save_thumbnail(lattice_images[0][slc], 'lattice-plastic.png')
save_thumbnail(lattice_images[1][slc], 'lattice-metal.png')
save_thumbnail(res_images[0][slc], 'lattice-plastic-high-res.png')

save_thumbnail(ia_maps[0][slc], 'ia-map.png', cmap=CMAP['artifact'], vmin=-0.6, vmax=0.6, mask=~implant_mask[slc])
save_thumbnail(gd_maps[0][..., 0][slc], 'gd-map.png', cmap=CMAP['distortion'], vmin=-2, vmax=2, mask=gd_masks[1][slc])
# save_thumbnail(snr_maps[0][slc], 'snr-map.png', cmap=CMAP['snr'], vmax=100, mask=snr_masks[0][slc])
save_thumbnail(res_maps[0][..., 0][res_slc], 'res-map.png', cmap=CMAP['resolution'], vmin=2, vmax=3) # TODO overlay mask, but need shapes to match first :(
