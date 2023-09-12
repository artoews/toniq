import matplotlib.pyplot as plt
import numpy as np

class MultiIndexTracker:
    """ Scroll through a collection of volumes in lock step """
    def __init__(self, fig, axes, volumes, plot_args):
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
        self.update(replot=True)
    
    @property
    def shape(self):
        return self.volumes[0].shape
    
    @property
    def max_index(self):
        return self.shape[-1] - 1
    
    def plot(self):
        self.ims = [
            ax.imshow(vol[:, :, self.index], **plot_args)
            for ax, vol, plot_args in zip(self.axes, self.volumes, self.plot_args)
            ]
    
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
                ax.clear()
            self.plot()
        else:
            for im, vol in zip(self.ims, self.volumes):
                im.set_data(vol[:, :, self.index])
        self.fig.suptitle(f'Press "r" to reformat \n Use scroll wheel to navigate \n At slice index {self.index}/{self.max_index}')
        if self.reformat:
            self.fig.supxlabel('Slice dimension (z)')
        else:
            self.fig.supxlabel('Phase encode dimension (y)')
        self.fig.canvas.draw()

def plotVolumes(volumes, nrows, ncols, vmin=0, vmax=1, cmap='gray', titles=None, figsize=None):
    if nrows * ncols != len(volumes):
        raise ValueError(
            'Number of volumes ({}) must equal number of subplots ({}x{})'
            .format(len(volumes), nrows, ncols)
            )
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=figsize)
    fig.supylabel('Readout direction (x)')
    if titles is not None:
        for ax, title in zip(axes.flatten(), titles):
            ax.set_title(title)
    plot_args = [{'vmin': vmin,
                 'vmax': vmax,
                 'cmap': cmap}
                 for _ in axes.flatten()]
    tracker = MultiIndexTracker(fig, axes.flatten(), volumes, plot_args)

    fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    fig.canvas.mpl_connect('key_press_event', tracker.on_press)
    return fig, tracker