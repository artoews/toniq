"""Functions for generic image filtering.

"""
import itertools
import numpy.typing as npt
import numpy as np

from multiprocessing import Pool
from typing import Callable

def batch_for_filter(
        arr: npt.NDArray,
        filter_size: int,
        axis: int,
        num_batches: int
        ) -> list[npt.NDArray]:
    """ Split array into batches along axis. """
    starts = np.arange(0, arr.shape[axis] - filter_size + 1)
    if len(starts) < num_batches:
        print('Warning: batching into {} sections cannot be done with only {} positions. Returning {} batches instead.'.format(num_batches, len(starts), len(starts)))
        num_batches = len(starts)
    batches = []
    for batch_starts in np.array_split(starts, num_batches):
        batch_indices = np.arange(batch_starts[0], batch_starts[-1] + filter_size)
        batch_data = np.take(arr, tuple(batch_indices), axis=axis)
        batches.append(batch_data)
    return batches

def generic_filter(
        image: npt.NDArray,
        function: Callable[[npt.NDArray], npt.NDArray],
        filter_shape: tuple[int],
        out_shape: tuple[int],
        stride: int,
        batch_axis: int,
        num_batches: int = 1
        ) -> npt.NDArray:
    """ Strided, batched filter implementing a user-passed filter function. """
    if num_batches > 1:
        image_batches = batch_for_filter(image, filter_shape[batch_axis], batch_axis, num_batches)
        num_batches = len(image_batches)
        inputs = list(zip(
            image_batches,
            (function,) * num_batches,
            (filter_shape,) * num_batches,
            (out_shape,) * num_batches,
            (stride,) * num_batches,
            (batch_axis,) * num_batches
            ))
        with Pool(num_batches) as p:
            result = p.starmap(generic_filter, inputs)
        return np.concatenate(result, axis=batch_axis)
    else:
        strides = np.roll((1, stride, stride), batch_axis)  # only stride the non-batch axes
        patch_locs = tuple(np.arange(0, image.shape[i] - filter_shape[i] + 1, strides[i]) for i in range(3))
        output = np.zeros(tuple(len(locs) for locs in patch_locs) + out_shape)
        for loc in itertools.product(*patch_locs):
            slc = tuple(slice(loc[i], loc[i] + filter_shape[i]) for i in range(3))
            idx = tuple(loc[i] // strides[i] for i in range(3))
            patch = image[slc]
            if np.isnan(patch).any():
                output[idx] = np.zeros(out_shape)
            else:
                output[idx] = function(patch)
        return output
