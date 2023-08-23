from dicom import load_series
from pathlib import Path

from pprint import pprint

series_dir = '/bmrNAS/people/artoews/data/scans/230801/13295_dicom/Series3'
# series_dir = '/bmrNAS/people/artoews/data/scans/230801/13295_dicom/Series7'

series_files = Path(series_dir).glob('*MRDC*')
d = load_series(series_files)

pprint(d)
print('Image matrix', d.shape, d.dtype)
print('Isotropic:', d.meta.isotropic)
print('Full RBW in kHz:', d.meta.readoutBandwidth_kHz)