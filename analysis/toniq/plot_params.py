"""Global parameters used for plotting.

"""

import matplotlib.pyplot as plt

SMALLER_SIZE = 6
SMALL_SIZE = 7
MEDIUM_SIZE = 8
LARGE_SIZE = 10
plt.rc('font', size=MEDIUM_SIZE, family='serif')         # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the x tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the y tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)   # fontsize of the figure title

plt.rc('lines', linewidth=0.7)
plt.rc('axes', linewidth=0.7)
plt.rcParams['hatch.linewidth'] = 0.2

# colormaps
CMAP = {
    'image': 'gray',
    'field': 'RdBu_r',
    'artifact': 'RdBu_r',
    'distortion': 'RdBu_r',
    'snr': 'viridis',
    'resolution': 'viridis'
}

DPI = 600

FIG_WIDTH = (3.42, 5.12, 6.9)  # MRM figure widths (in inches) for single-column, 1.5 column, double column