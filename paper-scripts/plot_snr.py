import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns

from os import path

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
styles = ['dotted', 'solid', 'dashed']

def setup_axes(ax, fontsize):
    loosely_dashed = (0, (5, 10))
    ax.axline((0, 0), (1, 1), color='k', linestyle=loosely_dashed)
    # ax.set_xlim([10, 55])
    # ax.set_ylim([10, 55])
    # ax.set_xticks(range(10, 51, 10))
    # ax.set_yticks(range(10, 51, 10))
    ax.set_xlabel('Expected SNR', fontsize=fontsize)
    ax.set_ylabel('Measured SNR', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.grid()

def scatter(snrs, rbw, save_dir=None, figsize=(8, 5), fontsize=20):
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.2)
    setup_axes(ax, fontsize*0.8)
    for i in range(1, len(rbw)):
        expected_factor = np.sqrt(rbw[0] / rbw[i])
        ax.scatter(expected_factor * snrs[0], snrs[i], c=colors[i-1], label='RBW={:.3g}kHz'.format(rbw[i]), s=0.01, marker='.')
        # ax.scatter(noise_stds[0] / expected_factor, noise_stds[i], c=colors[i-1], label='RBW={:.3g}kHz'.format(rbw[i]), s=0.01, marker='.')
    ax.legend(fontsize=fontsize)
    if save_dir is not None:
        fig.savefig(path.join(save_dir, 'snr_pixel_cloud.png'), dpi=300)
    return fig, ax

def lines(snrs, rbw, save_dir=None, figsize=(8, 5), fontsize=20):
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.2)
    setup_axes(ax, fontsize*0.8)
    for i in range(1, len(rbw)):
        expected_factor = np.sqrt(rbw[0] / rbw[i])
        expected_snr_rounded = np.round(expected_factor * snrs[0])
        # expected_noise_rounded = np.round(noise_stds[0] / expected_factor)
        # plots mean line and 95% confidence band
        sns.lineplot(x=expected_snr_rounded.ravel(), y=snrs[i].ravel(), ax=ax, legend='brief', label='RBW={:.3g}kHz'.format(rbw[i]), color=colors[i-1], linestyle=styles[i-1])
        # sns.lineplot(x=expected_noise_rounded.ravel(), y=noise_stds[i].ravel(), ax=ax3, legend='brief', label='{:.3g}kHz'.format(rbw[i]), color=colors[i-1])
    ax.legend(fontsize=fontsize)
    if save_dir is not None:
        fig.savefig(path.join(save_dir, 'validation_snr.png'), dpi=300)
    return fig, ax
