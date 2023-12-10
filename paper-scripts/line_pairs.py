import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
from os import path
from pathlib import Path

import analysis
import dicom

def load_dicom_series(path):
    if path is None:
        return None
    files = Path(path).glob('*MRDC*')
    image = dicom.load_series(files)
    return image

exam_root = '/Users/artoews/root/data/mri/231021/13882_dicom/'
series_list = ['Series10', 'Series11', 'Series12', 'Series13']
save_dir = 'tmp/resolution'

# load data
images = []
for series_name in series_list:
    image = load_dicom_series(path.join(exam_root, series_name))
    images.append(image)
    print('Found DICOM series {}; loaded data with shape {}'.format(series_name, image.shape))

# extract relevant metadata and throw away the rest
shapes = np.stack([np.array(image.meta.acqMatrixShape) for image in images])

# hack: overwrite images to be copies of the reference with masked k-space 
k_full = sp.fft(images[0].data)
fullShape = k_full.shape
for i in range(1, len(images)):
    acqShape = images[i].meta.acqMatrixShape
    print('hacking image {} to be copy of reference image {} with k-space shape {}'.format(i, fullShape, acqShape))
    noise = np.random.normal(size=acqShape, scale=8e2)
    k = sp.resize(sp.resize(k_full, acqShape) + noise, fullShape)
    # k = sp.resize(sp.resize(k_full, acqShape), fullShape)
    # k = sp.resize(sp.fft(images[i].data), fullShape)  # or just zero-pad original data to match reference
    images[i].data = np.abs(sp.ifft(k))

images = np.stack([image.data for image in images])

# rescale data for comparison
images[0] = analysis.normalize(images[0])
for i in range(1, len(images)):
    images[i] = analysis.equalize(images[i], images[0])

slc_x = (slice(182*2, 214*2), slice(113*2, 145*2), 15)
slc_y = (slice(182*2, 214*2), slice(151*2, 183*2), 15)
fig, axes = plt.subplots(nrows=2, ncols=len(images)-1, figsize=(12, 5))
# if len(images) == 2:
#     axes_x = np.array([axes_x])
#     axes_y = np.array([axes_y])
plot_kwargs = {'vmin': 0, 'vmax': 1, 'cmap': 'gray'}
for i in range(len(images)-1):
    j = i + 1
    print(images.shape)
    print(images[j].shape)
    print(slc_x)
    print(images[j][slc_x].shape)
    axes[0, i].imshow(images[j][slc_x], **plot_kwargs)
    axes[0, i].set_title('{} x {}\n'.format(shapes[j][0], shapes[j][1]), fontsize=16)
    axes[0, i].set_xlabel('{:.1f}'.format(shapes[0][0] / shapes[1+i][0]), fontsize=20)
    axes[1, i].imshow(images[j][slc_y], **plot_kwargs)
    axes[1, i].set_xlabel('{:.1f}'.format(shapes[0][1] / shapes[1+i][1]), fontsize=20)
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])
    axes[1, i].set_xticks([])
    axes[1, i].set_yticks([])
axes[0, 0].set_ylabel('X Line Pairs', fontsize=20)
axes[1, 0].set_ylabel('Y Line Pairs', fontsize=20)
plt.tight_layout()
plt.show()
fig.savefig(path.join(save_dir, 'line_pairs.png'), dpi=300)
