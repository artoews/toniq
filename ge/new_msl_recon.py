import numpy as np
import matplotlib.pyplot as plt

from GERecon import Archive, Cartesian3DAcquiredData, ZTransform

def cartesian_3d_recon(archive_path):

    # Initialize all required processing objects
    archive = Archive(archive_path)
    # zTransform = ZTransform(archive.ZTransformParams())

    # Initialize object for raw data access
    data = Cartesian3DAcquiredData(archive)

    # Gather information required to perform the reconstruction
    metadata = archive.Metadata()
    image_x_res = metadata["imageXRes"]
    image_y_res = metadata["imageYRes"]
    num_passes = metadata["passes"]
    slices_per_pass = archive.SlicesPerPass()
    num_echoes = data.NumEchoes()

    print('slices per pass', slices_per_pass)

    # Reconstruct all images in the scan
    # geometric_slice_indexes = np.zeros([max(slices_per_pass), num_passes], dtype=np.int)
    # volume_data = np.zeros([image_x_res, image_y_res, max(slices_per_pass), num_passes])
    # final_images = np.zeros([image_x_res, image_y_res, num_echoes, max(slices_per_pass), num_passes])
    num_coils = 30
    num_bins = num_passes * num_echoes
    kspace_data = np.zeros([image_x_res, image_y_res, num_coils, slices_per_pass, num_bins])

    for pass_number in range(num_passes):
        print('pass {} of {}'.format(pass_number, num_passes))
        for echo_number in range(num_echoes):
            print('echo {} of {}'.format(echo_number, num_echoes))
            kspace = data.GetVolume(pass_number, echo_number)
            print(kspace.shape, kspace.dtype)
            # num_channels = kspace.shape[2]
            # Run ZTransform on kspace data
            # for channel in range(num_channels):
            #     kspace[:, :, channel, :] = zTransform.Execute(kspace[:, :, channel, :])


if __name__ == '__main__':
    # file = '/Users/artoews/root/data/mri/231021/Series20/ScanArchive_415723SHMR18_20231021_205205589.h5'
    file = '/Users/artoews/root/data/mri/231021/Series21/ScanArchive_415723SHMR18_20231021_210028849.h5'
    cartesian_3d_recon(file)