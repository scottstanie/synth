import logging
from collections.abc import Callable, Mapping
from concurrent.futures import Executor, Future
from functools import partial

import h5py
import jax.numpy as jnp
import numpy as np
from jax import jit, lax

from ._types import P, PathOrStr, T

logger = logging.getLogger("synth")

ALL_LAYERS = slice(None)


def load_current_phase(
    files: Mapping[str, PathOrStr],
    rows: slice,
    cols: slice,
    idx: slice | int = ALL_LAYERS,
) -> np.ndarray:
    """Load and sum the phase data from multiple HDF5 files for a row/column block.

    Parameters
    ----------
    files : dict[str, Path]
        Dictionary of file paths for different phase components.
    rows : slice
        Row slice to extract.
    cols : slice
        Column slice to extract.
    idx : slice | int, optional
        Single index or slice of the 3D cube to load.
        Default is to load all depth layers.

    Returns
    -------
    np.ndarray: 3D Float32 array representing the summed true input phase data
    for the specified block.

    """
    summed_phase = None

    # logger.debug(f"Loading {files} at {idx=}, {rows=}, {cols=}")
    for _, file_path in files.items():
        with h5py.File(file_path, "r") as f:
            # Assume the main dataset is named 'data'. Adjust if necessary.
            dset: h5py.Dataset = f["data"]

            # Check if the dset is 3D
            if dset.ndim == 3:
                # For 3D datasets, load the full depth
                data = dset[idx, rows, cols]
            elif dset.ndim == 2:
                # For 2D datasets, add a depth dimension of 1
                data = dset[rows, cols][np.newaxis, :, :]
            else:
                raise ValueError(f"Unexpected dset shape in {file_path}: {dset.shape}")

            if summed_phase is None:
                summed_phase = data
            else:
                summed_phase += data

    if summed_phase is None:
        raise ValueError("No valid data found in the provided files.")

    return summed_phase


def _setup_logging(level=logging.INFO):
    if not logger.handlers:
        logger.setLevel(level)
        h = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        h.setFormatter(formatter)
        logger.addHandler(h)


def round_mantissa(z: np.ndarray, significant_bits=10, truncate: bool = False):
    """Zero out bits in mantissa of elements of array in place.

    Attempts to round the floating point numbers zeroing.

    Parameters
    ----------
    z : numpy.array
        Real or complex array whose mantissas are to be zeroed out
    significant_bits : int, optional
        Number of bits to preserve in mantissa. Defaults to 10.
        Lower numbers will truncate the mantissa more and enable
        more compression.
    truncate : bool, optional
        Instead of attempting to round, simply truncate the mantissa.
        Default = False

    """
    # recurse for complex data
    if np.iscomplexobj(z):
        round_mantissa(z.real, significant_bits)
        round_mantissa(z.imag, significant_bits)
        return

    if not issubclass(z.dtype.type, np.floating):
        err_str = "argument z is not complex float or float type"
        raise TypeError(err_str)

    mant_bits = np.finfo(z.dtype).nmant
    float_bytes = z.dtype.itemsize

    if significant_bits == mant_bits:
        return

    if not 0 < significant_bits <= mant_bits:
        err_str = f"Require 0 < {significant_bits=} <= {mant_bits}"
        raise ValueError(err_str)

    # create integer value whose binary representation is one for all bits in
    # the floating point type.
    allbits = (1 << (float_bytes * 8)) - 1

    # Construct bit mask by left shifting by nzero_bits and then truncate.
    # This works because IEEE 754 specifies that bit order is sign, then
    # exponent, then mantissa.  So we zero out the least significant mantissa
    # bits when we AND with this mask.
    nzero_bits = mant_bits - significant_bits
    bitmask = (allbits << nzero_bits) & allbits

    utype = np.dtype(f"u{float_bytes}")
    # view as uint type (can not mask against float)
    u = z.view(utype)

    if truncate is False:
        round_mask = 1 << (nzero_bits - 1)
        u += round_mask  # Add the rounding mask before applying the bitmask
    # bitwise-and in-place to mask
    u &= bitmask


class DummyProcessPoolExecutor(Executor):
    """Dummy ProcessPoolExecutor for to avoid forking for single_job purposes."""

    def __init__(self, max_workers: int | None = None, **kwargs):  # noqa: D107
        self._max_workers = max_workers

    def submit(  # noqa: D102
        self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        future: Future = Future()
        result = fn(*args, **kwargs)
        future.set_result(result)
        return future

    def shutdown(self, wait: bool = True, cancel_futures: bool = True):  # noqa: D102
        pass


@partial(jit, static_argnums=(1, 2, 3))
def take_looks(image, row_looks, col_looks, average=True):
    # Ensure the image has a channel/batch dimension (assuming grayscale image)
    # Add a (batch, ..., channel) dimensions to make NHWC
    image = image[jnp.newaxis, ..., jnp.newaxis]

    # Create a kernel filled with ones
    # Kernel shape: HWIO (height, width, input_channels, output_channels)
    kernel = jnp.ones((row_looks, col_looks, 1, 1), dtype=image.dtype)

    # With each window, we're jumping over by the same number of pixels
    strides = (row_looks, col_looks)
    result = lax.conv_general_dilated(
        image,
        kernel,
        window_strides=strides,
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )

    # Average if required
    if average:
        result /= row_looks * col_looks

    return result.squeeze()
