from dataclasses import dataclass
from enum import Enum
import numpy as np
from typing import ClassVar
from pydicom import dcmread

class PulseSequence(Enum):
    FSE = 1
    MAVRICSL = 2

class Plane(Enum):
    CORONAL = 1
    SAGITTAL = 2
    AXIAL = 3

@dataclass
class ImageVolume:
    """ Data class for each acquired image volume. """
    data: np.ndarray
    fieldOfView_mm: tuple
    readoutBandwidthFull_Hz: float
    containsMetal: bool
    pulseSequence: PulseSequence
    imagePlane: Plane
    NDIMS: ClassVar[int] = 3

    def __post_init__(self):
        if self.data.ndims != self.NDIMS:
            raise ValueError(
                'Invalid number of dimensions for data: got {}, require {}.'.format(self.data.ndims, self.NDIMS)
                )
        if len(self.fieldOfView_mm) != self.NDIMS:
            raise ValueError(
                'Invalid number of dimensions for FOV: got {}, require {}.'.format(len(self.fieldOfView_mm), self.NDIMS)
            )
        self.shape = self.data.shape
        self.resolution_mm = tuple(self.fieldOfView_mm[i] / self.shape[i] for i in range(self.NDIMS))

def dicom_to_numpy(files, dtype=None, slice_axis=-1):
    """ Convert a DICOM series representing an image volume into a numpy array """
    slices = [dcmread(f).pixel_array for f in files]
    volume = np.stack(slices, axis=slice_axis, dtype=dtype)
    return volume

# TODO parse DICOM to get FOV, RBW, pulse sequence, image plane?
# TODO establish a keyword in series name to indicate metal vs pla: use PLA to indicate a plastic replica, otherwise assume metal