import numpy as np
import matplotlib.pyplot as plt

from plot import plotVolumes
from util import equalize

b1 = np.load('sandbox/bin_1.npy')
b3 = np.load('sandbox/bin_3.npy')
b5 = np.load('sandbox/bin_5.npy')
b24 = np.load('sandbox/bin_24.npy')

# b1 = b1 / np.max(b1)
# b3 = b3 / np.max(b3)
# b5 = b5 / np.max(b5)
# b24 = b24 / np.max(b24)

(b24, b5, b3, b1), otsu_thresholds, signal_masks = equalize([b24, b5, b3, b1])
fig, tracker = plotVolumes((b24, b5, b3, b1) + tuple(signal_masks), nrows=2, ncols=4, titles=('24', '5', '3', '1'))
plt.show()