"""Functions for loading & parsing DICOM data.

"""
import numpy.typing as npt
import numpy as np

from pathlib import Path
from pydicom import dcmread
from os import path

from toniq.data import Metadata, ImageVolume

def load_series(files: list[str], dtype=float) -> ImageVolume:
    """ Load DICOM series a list of files. """
    data = []
    slice_indices = []
    meta = None
    for f in files:
        slice_data = read_data(f)
        slice_meta, slice_index = read_meta(f)
        if meta is None:
            meta = slice_meta
        elif meta != slice_meta:
            raise ValueError('Metadata disagreement between slices')
        data.append(slice_data)
        slice_indices.append(slice_index)
    order = np.argsort(slice_indices)
    data = np.stack([data[i] for i in order], axis=-1)
    data = data.astype(dtype)
    return ImageVolume(data, meta)

def load_series_from_path(series_path: str) -> ImageVolume:
    """ Load DICOM series located on a given path. """
    files = Path(series_path).glob('*MRDC*')
    image = load_series(files)
    return image

def read_data(file: str) -> npt.NDArray:
    """ Read image data from DICOM file. """
    return dcmread(file).pixel_array

def read_meta(file: str) -> tuple[Metadata, int]:
    """ Read metadata from DICOM file. """
    dicom = dcmread(file)
    acqMatrix = np.array(dicom.AcquisitionMatrix)
    inplaneMatrixSize = acqMatrix[np.nonzero(acqMatrix)]

    meta_dict = {
        'date_YYYYMMDD': dicom.StudyDate,
        'scanner': dicom.ManufacturerModelName,
        'staticFieldStrength_T': float(dicom.MagneticFieldStrength),

        'seriesName': dicom.SeriesDescription,
        'dimensionality': dicom.MRAcquisitionType,
        'pulseSequenceName': dicom[0x0019, 0x109c].value,
        'duration_s': np.round(dicom[0x0019, 0x105a].value * 1e-6),

        'acqMatrixShape': tuple(inplaneMatrixSize) + (int(dicom.ImagesInAcquisition),),
        'resolution_mm': tuple(map(float, dicom.PixelSpacing)) + (float(dicom.SliceThickness),),
        'refocusFlipAngle_deg': float(dicom.FlipAngle),
        'echoTrainLength': int(dicom.EchoTrainLength),
        'echoTime_ms': float(dicom.EchoTime),
        'repetitionTime_ms': float(dicom.RepetitionTime),
        'centerFrequency_Hz': float(dicom.ImagingFrequency * 1e6),
        'pixelBandwidth_Hz': float(dicom.PixelBandwidth),
        'readoutDirection': 0 if dicom.InPlanePhaseEncodingDirection == 'ROW' else 1,

        'containsMetal': 'PLA' not in dicom.SeriesDescription
    }
    
    if meta_dict['dimensionality'] == '2D' and dicom.SpacingBetweenSlices != meta_dict['resolution_mm'][-1]:
        raise ValueError('Multi-slice volume has gaps between slices')
    meta = Metadata(**meta_dict)
    try:
        sliceIndex = dicom.InStackPositionNumber
    except:
        _, sliceIndex = path.splitext(file)
        sliceIndex = int(sliceIndex[1:])
    return meta, sliceIndex
