import matplotlib.pyplot as plt

SMALL_SIZE = 10
MEDIUM_SIZE = 12
LARGE_SIZE = 14
plt.rc('font', size=MEDIUM_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=LARGE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the x tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the y tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)   # fontsize of the figure title

plt.rc('lines', linewidth=2.0)

# colormaps
CMAP = {
    'image': 'gray',
    'field': 'RdBu_r',
    'artifact': 'RdBu_r',
    'distortion': 'RdBu_r',
    'snr': 'viridis',
    'resolution': 'viridis'
}
