try:
    import matlab.engine as me
    from matlab import int8 as matlab_int8
except:
    print('Warning: matlab interface could not be imported.')

import numpy as np
from os import path
from scipy.io import loadmat
import yaml


class ScanArchive:
    def __init__(self, archive_file, verbose=False):
        if verbose:
            print('Opening ScanArchive at {}'.format(archive_file))
        self.file = archive_file
        self.meta = self.load_meta()

    def load_meta(self):
        """ Load metadata into a dictionary. """
        eng = me.start_matlab()
        meta_file = eng.metadata_to_mat(self.file)  # creates archive.mat
        meta = loadmat(meta_file, simplify_cells=True)
        return meta

    @property
    def image(self):
        """ Shortcut to 'image' subdirectory of metadata """
        return self.meta['DownloadData']['rdb_hdr_image']

    @property
    def rec(self):
        """ Shortcut to 'rec' subdirectory of metadata """
        return self.meta['DownloadData']['rdb_hdr_rec']

    def extract_params(self):

        rx = self.image['pixelSizeX']
        ry = self.image['pixelSizeY']
        rz = self.image['slthick']

        # currently unused
        fermi_radius = self.image['fermi_radius']
        fermi_width = self.image['fermi_width']
        fermi_ecc = self.image['fermi_ecc']
        hnover = self.rec['rdb_hdr_hnover']  # number of points over the halfway mark, when PF is active?? if 0, no PF
        rbw = self.rec['rdb_hdr_bw'] * 2

        p = {'shape': [self.nx, self.ny, self.nz],
             'fov': [self.nx * rx, self.ny * ry, self.nz * rz],
             'rbw': rbw,
             'num_coils': self.nc,
             'vat': True,
             }

        savefile = path.join(path.dirname(self.file), 'seq.yml')
        with open(savefile, 'w') as f:
            yaml.dump(p, f, default_flow_style=None)

    @property
    def nx(self):
        """ Size of x dimension. """
        return self.rec['rdb_hdr_da_xres']

    @property
    def ny(self):
        """ Size of y dimension. """
        return self.rec['rdb_hdr_da_yres'] - 1

    @property
    def nz(self):
        """ Size of z dimension. """
        return self.meta['SlicesPerPass']

    @property
    def nb(self):
        """ Size of bin dimension. """
        num_echoes = self.rec['rdb_hdr_nechoes']
        num_passes = self.rec['rdb_hdr_npasses']
        return num_echoes * num_passes

    @property
    def nc(self):
        """ Size of coil dimension. """
        stop = self.rec['rdb_hdr_dab'][0]['stop_rcv']
        start = self.rec['rdb_hdr_dab'][0]['start_rcv']
        return stop - start + 1

    @property
    def bin_order(self):
        """ Returns the index ordering of bins. """
        b0_offsets = self.rec['rdb_hdr_mavric_b0_offset']
        return np.argsort(b0_offsets[:self.nb]).argsort()

    def extract_data(self, flipread=False):
        eng = me.start_matlab()
        bin_order = (self.bin_order + 1).tolist()
        bin_order = matlab_int8(bin_order)
        eng.extract_mavricsl(self.file, self.nx, self.ny, self.nz, self.nb, self.nc, bin_order, flipread, nargout=0)  # takes ~5 minutes


if __name__ == '__main__':
    file = '/dataNAS/people/artoews/data/scans/220121/r2pf/read0/ScanArchive_415723SHMR18_20220121_180436741.h5'
    s = ScanArchive(file)
    print('nx', s.nx)
    print('ny', s.ny)
    print('nz', s.nz)
    s.extract_params()
    s.extract_data()
