from dataclasses import dataclass
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
    totalDuration_s: float

    matrixShape: tuple[int, int, int]
    resolution_mm: tuple[float, float, float]
    refocusFlipAngle_deg: float
    echoTrainLength: int
    echoTime_ms: float
    repetitionTime_ms: float
    centerFrequency_MHz: float
    pixelBandwidth_Hz: float
    phaseEncodingDirection: int

    containsMetal: bool

@dataclass
class ImageVolume:
    """ Data class for each acquired image volume. """
    data: np.ndarray
    meta: Metadata
    NDIMS: ClassVar[int] = 3

    def __post_init__(self):
        if self.data.ndims != self.NDIMS:
            raise ValueError(
                'Invalid number of dimensions for data: got {}, require {}.'.format(self.data.ndims, self.NDIMS)
                )
        self.shape = self.data.shape
        self.dtype = self.data.dtype
