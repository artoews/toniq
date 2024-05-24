from pydicom import dcmread

# Load test data
fpaths = [
    '/bmrNAS/people/artoews/data/scans/230801/13295_dicom/Series2/i46876168.MRDC.46',
    '/bmrNAS/people/artoews/data/scans/230801/13295_dicom/Series3/i46876252.MRDC.46',
    '/bmrNAS/people/artoews/data/scans/230801/13295_dicom/Series7/i46876612.MRDC.46',
    '/bmrNAS/people/artoews/data/scans/13275_dicom/Series6/i46114323.MRDC.6',
    '/bmrNAS/people/artoews/data/scans/13275_dicom/Series7/i46114337.MRDC.6',
    '/bmrNAS/people/artoews/data/scans/13275_dicom/Series9/i46114368.MRDC.6',
    '/bmrNAS/people/artoews/data/scans/13275_dicom/Series10/i46114382.MRDC.6',
]
datalist = [dcmread(f) for f in fpaths]

d = datalist[-1]

print(d)

print('MR Acquisition Type', d.MRAcquisitionType)
print('Pixel Spacing (mm)', d.PixelSpacing)
print('Internal Pulse Sequence Name', d[0x0019, 0x109e].value) # labels MAVRIC SL as 3DFSE
print('Pulse Sequence Name', d[0x0019, 0x109c].value) # product name
print('Echo Train Length', d.EchoTrainLength)
print('Echo Time (ms)', d.EchoTime)
print('Imaging Frequency (MHz)', d.ImagingFrequency)
print('Pixel Bandwidth (Hz)', d.PixelBandwidth)
print('Repetition Time (ms)', d.RepetitionTime)
print('Slice Spacing (mm)', d.SpacingBetweenSlices)
print('Slice Thickness (mm)', d.SliceThickness)
print('Magnetic Field Strength (T)', d.MagneticFieldStrength)
print('Study Date (YYYYMMDD)', d.StudyDate)
print('Series Description', d.SeriesDescription)
print('Model Name', d.ManufacturerModelName)
print('Refocussing Flip Angle (degrees)', d.FlipAngle)
print('Acquisition Matrix', d.AcquisitionMatrix)
print('In-plane phase encoding direction', d.InPlanePhaseEncodingDirection)
print('Acquisition Duration (s)', d[0x0019, 0x105a].value * 1e-6)
print('Locs per 3D slab', d[0x0021, 0x1057].value)
print('Image dimension - X (mm)', d[0x0027, 0x1060].value)
print('Image dimension - Y (mm)', d[0x0027, 0x1061].value)
print('In-Stack Position Number', d.InStackPositionNumber)
print('ImagesinAcquisition', d.ImagesInAcquisition)