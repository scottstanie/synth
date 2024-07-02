import logging

import h5py
import numpy as np

from ._types import PathOrStr

logger = logging.getLogger(__name__)


def load_current_phase(
    files: dict[str, PathOrStr], rows: slice, cols: slice
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

    Returns
    -------
    np.ndarray: 3D array representing the summed phase data for the specified block.

    """
    summed_phase = None

    for component, file_path in files.items():
        logger.debug(f"Loading {component}")
        with h5py.File(file_path, "r") as f:
            # Assume the main dataset is named 'data'. Adjust if necessary.
            dset: h5py.Dataset = f["data"]

            # Check if the dset is 3D
            if dset.ndim == 3:
                # For 3D datasets, load the full depth
                data = dset[:, rows, cols]
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


def _setup_logging():
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        h.setFormatter(formatter)
        logger.addHandler(h)
