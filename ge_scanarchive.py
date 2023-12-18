import numpy as np

from GERecon import Archive

class ScanArchive:
    def __init__(self, file):
        self.archive = Archive(file)

    @property
    def metadata(self):
        return self.archive.Metadata()

    @property
    def header(self):
        return self.archive.Header()
    
    @property
    def control_count(self):
        return self.metadata['controlCount']
    
    @property
    def readout_bandwidth(self):
        return self.header['rdb_hdr_rec']['rdb_hdr_bw'] * 2
    
    @property
    def num_coils(self):
        return self.metadata['numChannels']
        # stop = self.header['rdb_hdr_rec']['rdb_hdr_dab'][0]['stop_rcv']
        # start = self.header['rdb_hdr_rec']['rdb_hdr_dab'][0]['start_rcv']
        # return stop - start + 1
    
    def NextFrame(self):
        return self.archive.NextFrame()
    
    def NextControl(self):
        return self.archive.NextControl()

class MavricSL(ScanArchive):
    def __init__(self, file):
        super().__init__(file)

    @property
    def shape(self):
        return self.num_readout, self.num_pe, self.num_slices, self.num_bins, self.num_coils

    @property
    def num_readout(self):
        """ Size of readout (x) dimension. """
        return self.metadata['acquiredXRes']
        # return self.header['rdb_hdr_rec']['rdb_hdr_da_xres']

    @property
    def num_pe(self):
        """ Size of phase-encode (y) dimension. """
        ny = self.metadata['acquiredYRes']
        hnover = self.header['rdb_hdr_rec']['rdb_hdr_hnover']
        if hnover > 0:
            ny = 2 * (ny - hnover)
        return ny
        # return self.header['rdb_hdr_rec']['rdb_hdr_da_yres'] - 1

    @property
    def num_slices(self):
        """ Size of slice (z) dimension. """
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
    def num_bins(self):
        """ Size of bin dimension. """
        return int(self.num_echoes * self.num_passes)
    
    @property
    def num_echoes(self):
        return self.header['rdb_hdr_rec']['rdb_hdr_nechoes']
    
    @property
    def num_passes(self):
        # return self.header['rdb_hdr_rec']['rdb_hdr_npasses']
        return self.metadata['passes']

    def bin_order(self):
        """ Returns the index ordering of bins. """
        b0_offsets = self.header['rdb_hdr_rec']['rdb_hdr_mavric_b0_offset']
        return np.argsort(b0_offsets[:self.nb]).argsort()