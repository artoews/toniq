"""Classes for representing scan data. 

"""
from dataclasses import dataclass, field
import numpy as np
from typing import ClassVar


@dataclass(kw_only=True)
class Metadata:
    """ Data class for pulse sequence meta data. """

    date_YYYYMMDD: str
    scanner: str
    staticFieldStrength_T: float

    seriesName: str
    dimensionality: int
    pulseSequenceName: str
    duration_s: float

    acqMatrixShape: tuple[int, int, int]
    resolution_mm: tuple[float, float, float]
    refocusFlipAngle_deg: float
    echoTrainLength: int
    echoTime_ms: float
    repetitionTime_ms: float
    centerFrequency_Hz: float
    pixelBandwidth_Hz: float
    readoutDirection: int

    containsMetal: bool

    def __post_init__(self):
        resolution_rounded = np.round(self.resolution_mm, decimals=1)
        self.isotropic = np.all(resolution_rounded == resolution_rounded[0])
        self.readoutBandwidth_kHz = np.round(self.pixelBandwidth_Hz * self.acqMatrixShape[self.readoutDirection] * 1e-3, decimals=2)
    
@dataclass
class ImageVolume:
    """ Data class for each acquired image volume. """
    data: np.ndarray = field(repr=False)
    meta: Metadata
    NDIM: ClassVar[int] = 3

    def __post_init__(self):
        if self.data.ndim != self.NDIM:
            raise ValueError(
                'Invalid number of dimensions for data: got {}, require {}.'.format(self.data.ndim, self.NDIM)
                )
        self.shape = self.data.shape
        self.dtype = self.data.dtype
    
    @property
    def is_isotropic(self):
        return True if len(set(self.meta.resolution_mm)) else False
