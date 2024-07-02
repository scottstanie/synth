import argparse
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import rasterio as rio
import tqdm
from rasterio.windows import Window

from ._blocks import iter_blocks
from ._types import PathOrStr
from .utils import _setup_logging, load_current_phase

logger = logging.getLogger(__name__)

OUTPUT_PROFILE_DEFAULTS = {
    "driver": "GTiff",
    "dtype": "float32",
    "nodata": 0.0,
    "count": 1,
    "tiled": True,
    "compress": "lzw",
    "nbits": "16",
}
DEFAULT_TRUTH_FILES = {
    "deformation": Path("deformation.h5"),
    "ramps": Path("phase_ramps.h5"),
    "turbulence": Path("turbulence.h5"),
}
BLOCK_SHAPE = (512, 512)


def compare_to_deformation(
    unw_files: Iterable[PathOrStr],
    conncomp_files: Iterable[PathOrStr],
    truth_files: Mapping[str, PathOrStr] = DEFAULT_TRUTH_FILES,
    output_dir: Path = Path("differences"),
    exclude_zero_conncomps: bool = True,
) -> list[Path]:
    """Compare unwrapped interferograms to synthetic deformation.

    This function compares a set of unwrapped interferogram files to synthetic deformation
    data, optionally excluding areas with zero connected components, and outputs the
    differences as GeoTIFF files.

    Parameters
    ----------
    unw_files : Iterable[PathOrStr]
        An iterable of paths to unwrapped interferogram files.
    conncomp_files : Iterable[PathOrStr]
        An iterable of paths to connected component files corresponding to the unwrapped interferograms.
    deformation_file : PathOrStr, optional
        Path to the HDF5 file containing synthetic deformation data.
        Default is Path("input_layers/deformation.h5").
    output_dir : Path, optional
        Directory where the difference files will be saved. Default is Path("differences").
    exclude_zero_conncomps : bool, optional
        If True, areas with zero connected components will be excluded from the comparison.
        Default is True.

    Returns
    -------
    list[Path]
        A list of paths to the output difference files.

    Raises
    ------
    ValueError
        If the `len(unw_files)` doesn't match `len(conncomp_files)`
        If the `len(unw_files)` isn't one less than the number of deformation layers

    """
    unw_file_list = list(unw_files)
    conncomp_file_list = list(conncomp_files)

    if len(unw_file_list) != len(conncomp_file_list):
        raise ValueError(f"{len(unw_file_list) = }, but {len(conncomp_file_list) = }")

    with rio.open(unw_file_list[0]) as src:
        shape2d = src.shape
        profile = src.profile | OUTPUT_PROFILE_DEFAULTS

    output_dir.mkdir(exist_ok=True, parents=True)
    output_files = [output_dir / f"difference_{f.name}" for f in unw_file_list]
    # Set up all output files
    for filename in output_files:
        with rio.open(filename, "w", **profile) as dst:
            pass

    # TODO: get better organization using the filenames/dates to
    # ensure we difference things correctly
    b_iter = list(iter_blocks(arr_shape=shape2d, block_shape=BLOCK_SHAPE))
    for rows, cols in tqdm(b_iter):
        truth_phase = load_current_phase(truth_files, rows=rows, cols=cols)
        if len(unw_file_list) != (truth_phase.shape[0] - 1):
            raise ValueError(
                f"{len(unw_file_list) = } should be {truth_phase.shape[0] = } - 1"
            )
        window = Window.from_slices(rows, cols)

        for cur_truth, in_f, out_f, cc_f in zip(
            truth_phase, unw_file_list, output_files, conncomp_file_list
        ):
            with rio.open(in_f) as src:
                cur_unw = src.read(1, window=window)

            difference = cur_unw - cur_truth

            if exclude_zero_conncomps:
                with rio.open(cc_f) as src:
                    mask = src.read(1, window=window) == 0
            else:
                mask = np.zeros(cur_unw.shape, dtype=bool)
            difference[mask] = 0

            with rio.open(out_f, "w", **profile) as dst:
                dst.write(difference, indexes=1)

            for filename, layer in zip(output_files, difference):
                with rio.open(filename, "r+") as dst:
                    dst.write(layer, 1, window=window)
    return output_files


def _get_cli_args():
    parser = argparse.ArgumentParser(
        description="Compare unwrapped interferograms to synthetic deformation."
    )
    parser.add_argument(
        "--unw-files",
        nargs="+",
        type=Path,
        help="Paths to unwrapped interferogram files.",
    )
    parser.add_argument(
        "--conncomp-files",
        nargs="+",
        type=Path,
        help="Paths to connected component files",
    )
    parser.add_argument(
        "--input-layers-dir",
        type=Path,
        default=Path("input_layers/"),
        help="Path to the HDF5 files containing synthetic deformation data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("differences"),
        help="Directory where the difference files will be saved",
    )
    parser.add_argument(
        "--include-zero-conncomps",
        action="store_false",
        dest="exclude_zero_conncomps",
        help="Include areas with zero connected components in the comparison",
    )
    return parser.parse_args()


def main():
    """Run the comparison analysis on a set of output files."""
    _setup_logging()

    args = _get_cli_args()
    truth_files = {k: args.input_layers_dir / v for k, v in DEFAULT_TRUTH_FILES.items()}

    output_files = compare_to_deformation(
        unw_files=args.unw_files,
        conncomp_files=args.conncomp_files,
        truth_files=truth_files,
        output_dir=args.output_dir,
        exclude_zero_conncomps=args.exclude_zero_conncomps,
    )

    logger.info("Comparison completed. Output files:")
    for file in output_files:
        logger.info(f"  {file}")


if __name__ == "__main__":
    main()
