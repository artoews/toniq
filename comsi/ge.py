import sys
import numpy as np
import sigpy as sp
from os import path
import yaml
from time import time

from GERecon import Archive, Arc, ChannelCombiner, Gradwarp, Orientation, Transformer, Dicom
import matplotlib.pyplot as plt

from plot import plotVolumes
from util import debug

# TODO check all properties are correct
# TODO explore self.archive for more

def flatten(kspace, verbose=True):
    if verbose:
        t0 = debug('start')
    init_shape = kspace.shape
    nx, ny, nz, nb, nc = init_shape
    kspace = kspace[:, mask, :]
    if verbose:
        debug('done masking shape {} to shape {}'.format(init_shape, kspace.shape), t0)
    init_shape = kspace.shape
    kspace = np.reshape(kspace, (nx, -1, nb, nc))
    if verbose:
        debug('done reshaping from {} to {}'.format(init_shape, kspace.shape), t0)
    init_shape = kspace.shape
    kspace = np.reshape(kspace, (-1, nb, nc))  # this part is really slow
    if verbose:
        debug('done reshaping from {} to {}'.format(init_shape, kspace.shape), t0)
    return kspace


class ScanArchive:
    def __init__(self, file, verbose=False):
        if verbose:
            print('Opening ScanArchive at {}'.format(file))
        self.archive = Archive(file)

    @property
    def metadata(self):
        return self.archive.Metadata()

    @property
    def header(self):
        return self.archive.Header()

    @property
    def shape(self):
        return self.nx, self.ny, self.nz, self.nb, self.nc

    @property
    def rbw(self):
        """ Readout bandwidth (kHz) """
        return self.header['rdb_hdr_rec']['rdb_hdr_bw'] * 2

    @property
    def nx(self):
        """ Size of x dimension. """
        return self.metadata['acquiredXRes']
        # return self.header['rdb_hdr_rec']['rdb_hdr_da_xres']

    @property
    def ny(self):
        """ Size of y dimension. """
        ny = self.metadata['acquiredYRes']
        hnover = self.header['rdb_hdr_rec']['rdb_hdr_hnover']
        if hnover > 0:
            ny = 2 * (ny - hnover)
        return ny
        # return self.header['rdb_hdr_rec']['rdb_hdr_da_yres'] - 1

    @property
    def nz(self):
        """ Size of z dimension. """
        # return self.metadata['SlicesPerPass']
        return int(self.header['rdb_hdr_image']['slquant'])
        # return self.meta['SlicesPerPass']

    @property
    def pff(self):
        """ Partial Fourier fraction. """
        hnover = self.header['rdb_hdr_rec']['rdb_hdr_hnover']
        if hnover > 0:
            return hnover / self.ny
        else:
            return 1

    @property
    def nb(self):
        """ Size of bin dimension. """
        num_echoes = self.header['rdb_hdr_rec']['rdb_hdr_nechoes']
        num_passes = self.metadata['passes']
        # num_passes = self.header['rdb_hdr_rec']['rdb_hdr_npasses']
        return int(num_echoes * num_passes)

    @property
    def nc(self):
        """ Size of coil dimension. """
        return self.metadata['numChannels']
        # stop = self.header['rdb_hdr_rec']['rdb_hdr_dab'][0]['stop_rcv']
        # start = self.header['rdb_hdr_rec']['rdb_hdr_dab'][0]['start_rcv']
        # return stop - start + 1

    def bin_order(self):
        """ Returns the index ordering of bins. """
        b0_offsets = self.header['rdb_hdr_rec']['rdb_hdr_mavric_b0_offset']
        return np.argsort(b0_offsets[:self.nb]).argsort()

    def extract_data(self, flipread=False, verbose=True):

        if verbose:
            t0 = debug('start')

        num_passes = self.metadata['passes']
        num_control = self.metadata['controlCount']
        bins_per_pass = int(self.nb / num_passes)  # pre-computed to save time in for loop
        bin_order = self.bin_order()

        kspace = np.zeros(self.shape, dtype=np.complex64)
        mask = np.zeros(self.shape[1:4], dtype=bool)
        ip = 1

        control_table = np.zeros((num_control, 3), dtype=int)

        if verbose:
            debug('beginning control loop with {} iterations'.format(num_control), t0)
        for ic in range(num_control):  # philip has -1?
            control = self.archive.NextControl()
            if control['opcode'] == 1:
                data = np.squeeze(self.archive.NextFrame())
                while data.shape != (self.nx, self.nc):
                    print('at control {}: skipping frame for bad shape; got {}, expected {}'.format(ic, data.shape, (self.nx, self.nc)))
                    data = np.squeeze(self.archive.NextFrame())
                iy = control['viewNum'] - 1
                iz = control['sliceNum']
                # iz_alternate = self.archive.Info(control['sliceNum'], ip)['sliceNumber']
                # assert iz == iz_alternate, 'iz and iz_alternate disagree {} {}'.format(iz, iz_alternate)
                ib = bin_order[control['echoNum'] + (ip - 1) * bins_per_pass]
                if np.mod(iz, 2) == 0:
                    data = -data  # half-FOV shift in z - why is this necessary?
                kspace[:, iy, iz, ib, :] = data
                mask[iy, iz, ib] = True
                control_table[ic, :] = [ib, iz, iy]
            elif control['opcode'] == 0:
                if ip < num_passes:
                    debug('incrementing pass index')
                    ip += 1
            if verbose and np.mod(ic, 10000) == 0:
                debug('control {}'.format(ic), t0)
        if verbose:
            debug('done control loop', t0)

        if flipread:
            kspace = np.flip(kspace, axis=0)
        print(kspace.shape)

        if verbose:
            debug('end', t0)
        return kspace, mask
    
    def save_params(self, file):
        rx = self.header['rdb_hdr_image']['pixelSizeX']
        ry = self.header['rdb_hdr_image']['pixelSizeY']
        rz = self.header['rdb_hdr_image']['slthick']

        # currently unused
        fermi_radius = self.header['rdb_hdr_image']['fermi_radius']
        fermi_width = self.header['rdb_hdr_image']['fermi_width']
        fermi_ecc = self.header['rdb_hdr_image']['fermi_ecc']

        p = {'shape': [self.nx, self.ny, self.nz],
             'fov': [self.nx * rx, self.ny * ry, self.nz * rz],
             'rbw': self.rbw,
             'num_coils': self.nc,
             'vat': True,
             }

        with open(file, 'w') as f:
            yaml.dump(p, f, default_flow_style=None)

    def gradwarp(self, img, orient=False, isotropic=True):
        # for magnitude image with shape (x, y, z)
        img = np.abs(img).astype(np.float32)
        if isotropic:
            fov_scaling = {'FrequencyPixelScaling': 1.0, 'PhasePixelScaling': 1.0}
        else:
            fov_scaling = self.archive.GradwarpParams()
        corners = (self.archive.Corners(0), self.archive.Corners(self.nz-1))
        # TODO pass gradient coefficients instead, so you don't have to hardcode the gradient type
        gw_img = Gradwarp().Execute3D(img, corners, fov_scaling, gradient='HRMW')
        if orient:
            gw_img = self.orient(gw_img)
        return gw_img

    def orient(self, img):
        def pad_fov(img):
            metadata = self.archive.Metadata()
            target_shape = (metadata['imageXRes'], metadata['imageYRes'])
            px = (target_shape[0] - img.shape[0]) // 2
            py = (target_shape[1] - img.shape[1]) // 2
            return np.pad(img, ((px, px), (py, py), (0, 0)))
        img = pad_fov(img)
        for iz in range(self.nz):
            orientation = self.archive.Orientation(iz)  # requires target shape to run
            img[:, :, iz] = Orientation().OrientImage(img[:, :, iz], orientation)
        return img


if __name__ == '__main__':
    # file = '/bmrNAS/people/artoews/data/scans/221125/Series15/ScanArchive_415723SHMR18_20221125_143248777.h5'
    file = '/Users/artoews/root/data/mri/231021/Series20/ScanArchive_415723SHMR18_20231021_205205589.h5'
    s = ScanArchive(file, verbose=True)
    s.save_params('seq.yml')
    kspace, mask = s.extract_data(flipread=False)
    print(kspace.shape)
    np.save('kspace.npy', kspace)
    print('doing ifft')
    image = np.abs(sp.ifft(kspace, oshape=kspace.shape, axes=(0, 1, 2)))
    image = image / np.max(image) * 2
    print('done ifft')
    fig1, tracker1 = plotVolumes((image[:, :, 16, :, 15],))
    fig2, tracker2 = plotVolumes((image[:, :, 16, 12, :],))
    plt.show()
