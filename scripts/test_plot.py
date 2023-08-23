import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from pprint import pprint

from dicom import load_series
from plot import plotVolumes

series_dirs = [
    '/bmrNAS/people/artoews/data/scans/230801/13295_dicom/Series3',
    '/bmrNAS/people/artoews/data/scans/230801/13295_dicom/Series7'
] 

volumes = []
peak_val = 0
for dir in series_dirs:
    series_files = Path(dir).glob('*MRDC*')
    d = load_series(series_files)
    pprint(d)
    peak_val = max(peak_val, np.max(d.data))
    volumes.append(d.data)

fig, tracker = plotVolumes(volumes, 1, 2, vmax=peak_val, titles=('Series3', 'Series7'), figsize=(6, 6))
plt.show()

