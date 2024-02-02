import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from plot_params import *

class MultiIndexTracker:
    """ Scroll through a collection of volumes in lock step """
    def __init__(self, fig, axes, volumes, plot_args, cbar=False):
        for vol in volumes:
            if vol.shape != volumes[0].shape:
                raise ValueError(
                    'Volume shapes must all be equal. Found volumes with shapes {} and {}'
                    .format(vol.shape, volumes[0].shape))
        self.fig = fig
        self.axes = axes
        self.volumes = volumes
        self.index = self.max_index // 2
        self.reformat = False
        self.plot_args = plot_args
        self.cbar = cbar
        self.update(replot=True)
    
    @property
    def shape(self):
        return self.volumes[0].shape
    
    @property
    def max_index(self):
        if self.ndim == 3:
            return self.shape[2] - 1
        else:
            return 0
    
    @property
    def ndim(self):
        return self.volumes[0].ndim
    
    def plot(self):
        if self.ndim == 3:
            self.ims = [
                ax.imshow(vol[:, :, self.index], **plot_args)
                for ax, vol, plot_args in zip(self.axes, self.volumes, self.plot_args)
                ]
        elif self.ndim == 2:
            self.ims = [
                ax.imshow(vol, **plot_args)
                for ax, vol, plot_args in zip(self.axes, self.volumes, self.plot_args)
                ]
        if self.cbar:
            for im in self.ims:
                plt.colorbar(im)
            self.cbar = False
    
    def on_scroll(self, event):
        print(event.button, event.step)
        increment = 1 if event.button == 'up' else -1
        self.index += increment
        self.update(replot=False)
    
    def on_press(self, event):
        if event.key == 'r':  # reformat
            self.volumes = [np.swapaxes(vol, 1, 2) for vol in self.volumes]
            self.index = self.max_index // 2
            self.reformat = not self.reformat
        self.update(replot=True)

    def update(self, replot=False):
        self.index = np.clip(self.index, 0, self.max_index)
        if replot:
            for ax in self.axes:
                for im in ax.get_images():
                    im.remove()
                # ax.clear()
            self.plot()
        else:
            for im, vol in zip(self.ims, self.volumes):
                if self.ndim == 3:
                    im.set_data(vol[:, :, self.index])
                elif self.ndim == 2:
                    im.set_data(vol)
        self.fig.suptitle(f'Press "r" to reformat \n Use scroll wheel to navigate \n At slice index {self.index}/{self.max_index}')
        if self.reformat:
            self.fig.supxlabel('Slice dimension (z)')
        else:
            self.fig.supxlabel('Phase encode dimension (y)')
        self.fig.canvas.draw()

def plotVolumes(volumes, nrows=None, ncols=None, vmin=0, vmax=1, cmap='gray', titles=None, figsize=None, cbar=False):
    if nrows is None or ncols is None:
        nrows = 1
        ncols = len(volumes)
    if nrows * ncols != len(volumes):
        raise ValueError(
            'Number of volumes ({}) must equal number of subplots ({}x{})'
            .format(len(volumes), nrows, ncols)
            )
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=figsize)
    if nrows * ncols == 1:
        axes = np.array([axes])
    fig.supylabel('Readout direction (x)')
    if titles is not None:
        for ax, title in zip(axes.flatten(), titles):
            ax.set_title(title)
    plot_args = [{'vmin': vmin,
                 'vmax': vmax,
                 'cmap': cmap}
                 for _ in axes.flatten()]
    tracker = MultiIndexTracker(fig, axes.flatten(), volumes, plot_args, cbar=cbar)

    fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    fig.canvas.mpl_connect('key_press_event', tracker.on_press)
    return fig, tracker

def overlay_mask(ax, mask, color=[100, 100, 100], alpha=255):
    color_mask = np.zeros(mask.shape + (4,), dtype=np.uint8)
    color_mask[mask, :] = np.array(color + [alpha], dtype=np.uint8)
    ax.imshow(color_mask)
    return

def letter_annotation(ax, xoffset, yoffset, letter):
    try:
        ax.text(xoffset, yoffset, letter, transform=ax.transAxes, size=18, weight='bold')
    except:
        ax.text2D(xoffset, yoffset, letter, transform=ax.transAxes, size=18, weight='bold') # works when ax is Axes3D

def imshow2(ax, im, slc1, slc2, vmin=0, vmax=1, cmap='gray', mask=None, y_label=None, x1_label=None, x2_label=None):
    divider = make_axes_locatable(ax)
    im1 = im[slc1]
    im2 = im[slc2]
    ratio = im2.shape[1] / im1.shape[1] * 100
    ax2 = divider.append_axes("right", size=str(ratio) + "%", pad=0.1)
    fig1 = ax.get_figure()
    fig1.add_axes(ax2)
    im1= ax.imshow(im[slc1], vmin=vmin, vmax=vmax, cmap=cmap)
    im2 = ax2.imshow(im[slc2], vmin=vmin, vmax=vmax, cmap=cmap)
    if mask is not None:
        overlay_mask(ax, mask[slc1])
        overlay_mask(ax2, mask[slc2])
    ax2.yaxis.set_tick_params(labelleft=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    if y_label is not None:
        ax.set_ylabel(y_label)
    if x1_label is not None:
        ax.set_xlabel(x1_label)
    if x2_label is not None:
        ax2.set_xlabel(x2_label)
    return im1, im2

def readout_arrow_annotation(ax, xy=(0.5, 0.7), xytext=(0.5, 0.1), color='black'):
    ax.annotate(
        "readout",
        color='black',
        xy=xy,
        xytext=xytext,
        xycoords='axes fraction',
        verticalalignment='bottom',
        horizontalalignment='center',
        arrowprops=dict(width=2, headwidth=8, headlength=8, color=color)
    )
