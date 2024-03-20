import numpy as np

from ge_mavricsl import get_bins_from_archive, combine_bins, correct_geometry

def select_bins(image_xyzb, indices):
    for i in range(image_xyzb.shape[-1]):
        if i not in indices:
            image_xyzb[..., i] = 0
    return image_xyzb

# file = '/Users/artoews/root/data/mri/231021/Series21/ScanArchive_415723SHMR18_20231021_210028849.h5'
file = '/Users/artoews/root/data/mri/231021/Series25/ScanArchive_415723SHMR18_20231021_212355693.h5'
# file = '/Users/artoews/root/data/mri/231021/Series20/ScanArchive_415723SHMR18_20231021_205205589.h5'
image_xyzb, offsets = get_bins_from_archive(file)
np.save('image_xyzb', image_xyzb)
np.save('offsets', offsets)
# image_xyzb = np.load('image_xyzb.npy')
# offsets = np.load('offsets.npy')

bin_subsets = ((11,), (10, 11, 12), (9, 10, 11, 12, 13), tuple(range(24)))
names = ('bin2_1', 'bin2_3', 'bin2_5', 'bin2_24')

for i in range(len(names)):
    print('doing bin subset {}'.format(i))
    image_xyzb_subset = select_bins(image_xyzb.copy(), bin_subsets[i])
    image_xyz = combine_bins(image_xyzb_subset, offsets)
    image_xyz = correct_geometry(file, image_xyz)
    np.save(names[i], image_xyz)
