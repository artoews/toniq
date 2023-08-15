import numpy as np
from pydicom import dcmread

from data import Metadata, ImageVolume

def load_series(files, dtype=None, slice_axis=-1):
    """ Initialize an ImageVolume from a slice series of DICOM files """
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
    data = np.stack(data, axis=slice_axis, dtype=dtype)
    return ImageVolume(data, meta)

def read_data(file):
    """ Read image data from DICOM file """
    return dcmread(file).pixel_array

def read_meta(file):
    """ Read metadata from DICOM file """
    dicom = dcmread(file)
    meta_dict = {
        'date_YYYYMMDD': dicom.StudyDate,
        'scanner': dicom.ManufacturerModelName,
        'staticFieldStrength_T': dicom.MagneticFieldStrength,

        'seriesName': dicom.SeriesDescription,
        'dimensionality': dicom.MRAcquisitionType,
        'pulseSequenceName': dicom[0x0019, 0x109c].value,
        'totalDuration_s': dicom[0x0019, 0x105a].value * 1e-6,

        'matrixShape': dicom.AcquisitionMatrix + (dicom.ImagesInAcquisition,),
        'resolution_mm': dicom.PixelSpacing + (dicom.sliceThickness,),
        'sliceSpacing_mm': dicom.SpacingBetweenSlices,
        'refocusFlipAngle_deg': dicom.FlipAngle,
        'echoTrainLength': dicom.EchoTrainLength,
        'echoTime_ms': dicom.EchoTime,
        'repetitionTime_ms': dicom.RepetitionTime,
        'centerFrequency_MHz': dicom.ImagingFrequency,
        'pixelBandwidth_Hz': dicom.PixelBandwidth,
        'phaseEncodingDirection': 1 if dicom.InPlanePhaseEncodingDirection == 'ROW' else 0,

        'containsMetal': 'PLA' not in dicom.SeriesDescription
    }
    meta = Metadata(**meta_dict)
    sliceIndex = dicom.InStackPositionNumber
    return meta, sliceIndex
