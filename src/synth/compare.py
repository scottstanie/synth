import argparse
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import rasterio as rio
from rasterio.windows import Window
from tqdm.auto import tqdm

from ._blocks import iter_blocks
from ._types import PathOrStr
from .utils import _setup_logging, load_current_phase

logger = logging.getLogger(__name__)

OUTPUT_PROFILE_DEFAULTS = {
    "driver": "GTiff",
    "dtype": "float32",
    "nodata": np.nan,
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


def compare_phase(
    phase_files: Iterable[PathOrStr],
    truth_files: Mapping[str, PathOrStr] = DEFAULT_TRUTH_FILES,
    output_dir: Path = Path("differences"),
    is_wrapped: bool = False,
    temporal_coherence_file: Optional[PathOrStr] = None,
    temporal_coherence_threshold: float = 0.7,
    conncomp_files: Optional[Iterable[PathOrStr]] = None,
    exclude_zero_conncomps: bool = True,
    reference_idx: int = 0,
) -> list[Path]:
    """Compare unwrapped or wrapped interferograms to synthetic deformation.

    This function compares a set of interferograms to synthetic deformation
    data, optionally excluding areas with zero connected components or
    low temporal coherence, and outputs the differences as GeoTIFF files.

    Parameters
    ----------
    phase_files : Iterable[PathOrStr]
        An iterable of paths to wrapped, or unwrapped, interferogram files.
    truth_files : dict[str, PathOrStr], optional
        Path to the HDF5 file containing synthetic deformation data.
        Default is DEFAULT_TRUTH_FILES.
    output_dir : Path, optional
        Directory where the difference files will be saved.
        Default is Path("differences").
    is_wrapped : bool, optional
        Indicate if the input files are wrapped phase. Default is False.
    temporal_coherence_file: Optional[PathOrStr], optional
        Path to a temporal coherence file, used to ignore portions of the results with
        low coherence. Default is None.
    temporal_coherence_threshold : float, optional
        If `temporal_coherence_file` is passed, the threshold used to ignore
        low coherence areas.
        Default is 0.7.
    conncomp_files : Optional[Iterable[PathOrStr]], optional
        An iterable of paths to connected component files corresponding to the
        unwrapped interferograms, if comparing unwrapped results. Default is None.
    exclude_zero_conncomps : bool, optional
        If True, areas with connected component == 0 are excluded from the comparison.
        Default is True.
    reference_idx : int
        Index of reference phase. Each layer will subtract this index.
        For single reference interferograms, referenced to first date, this is 0.

    Returns
    -------
    list[Path]
        A list of paths to the output difference files.

    Raises
    ------
    ValueError
        If the number of phase files doesn't match the number of truth phase layers.

    """
    phase_file_list = list(phase_files)

    with rio.open(phase_file_list[0]) as src:
        shape2d = src.shape
        profile = src.profile | OUTPUT_PROFILE_DEFAULTS

    row_strides, col_strides = _get_downsample_factor(
        shape2d, truth_files["deformation"]
    )

    output_dir.mkdir(exist_ok=True, parents=True)
    output_files = [output_dir / f"difference_{Path(f).name}" for f in phase_file_list]

    for filename in output_files:
        with rio.open(filename, "w", **profile) as dst:
            pass

    b_iter = list(iter_blocks(arr_shape=shape2d, block_shape=BLOCK_SHAPE))
    for rows, cols in tqdm(b_iter):
        full_rows = slice(
            rows.start * row_strides, rows.stop * row_strides, row_strides
        )
        full_cols = slice(
            cols.start * col_strides, cols.stop * col_strides, col_strides
        )

        truth_phase = load_current_phase(truth_files, rows=full_rows, cols=full_cols)

        # TODO: do we always assume single ref, first date?
        # Should probably come up with a way to specify this...
        truth_phase = truth_phase[1:] - truth_phase[[reference_idx]]

        if len(phase_file_list) != truth_phase.shape[0]:
            raise ValueError(
                f"{len(phase_file_list) = } should be {truth_phase.shape[0] = }"
            )

        window = Window.from_slices(rows, cols)

        # Load temporal coherence mask if provided
        if temporal_coherence_file:
            with rio.open(temporal_coherence_file) as src:
                temp_coh = src.read(1, window=window)
            coh_mask = temp_coh < temporal_coherence_threshold

        for cur_truth, in_f, out_f in zip(truth_phase, phase_file_list, output_files):
            with rio.open(in_f) as src:
                cur_phase = src.read(1, window=window)

            if is_wrapped:
                difference = np.angle(np.exp(1j * (cur_phase - cur_truth)))
            else:
                difference = cur_phase + cur_truth
                difference -= difference.mean()

            # Apply masks
            if temporal_coherence_file:
                difference[coh_mask] = np.nan

            if conncomp_files and exclude_zero_conncomps:
                cc_f = next(conncomp_files)
                with rio.open(cc_f) as src:
                    mask = src.read(1, window=window) == 0
                difference[mask] = np.nan

            with rio.open(out_f, "r+") as dst:
                dst.write(difference, indexes=1, window=window)

    return output_files


def _get_downsample_factor(downsampled_shape, full_res_file):
    with h5py.File(full_res_file) as hf:
        full_shape = hf["data"].shape

    row_strides = int(round(full_shape[-2] / downsampled_shape[0]))
    col_strides = int(round(full_shape[-1] / downsampled_shape[1]))
    return row_strides, col_strides


def _get_cli_args():
    parser = argparse.ArgumentParser(
        description="Compare phase files to synthetic deformation."
    )
    parser.add_argument(
        "--phase-files",
        nargs="+",
        type=Path,
        required=True,
        help="Paths to phase files (wrapped or unwrapped).",
    )
    parser.add_argument(
        "--wrapped",
        action="store_true",
        help="Indicate if the input files are wrapped phase.",
    )
    parser.add_argument(
        "--conncomp-files",
        nargs="+",
        type=Path,
        help="Paths to connected component files (for unwrapped phase only)",
    )
    parser.add_argument(
        "--temporal-coherence-file",
        type=Path,
        help="Path to temporal coherence file",
    )
    parser.add_argument(
        "--temporal-coherence-threshold",
        type=float,
        default=0.7,
        help="Threshold for temporal coherence masking",
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
    """Run the comparison analysis on a set of phase files."""
    _setup_logging()

    args = _get_cli_args()
    truth_files = {k: args.input_layers_dir / v for k, v in DEFAULT_TRUTH_FILES.items()}

    output_files = compare_phase(
        phase_files=args.phase_files,
        truth_files=truth_files,
        output_dir=args.output_dir,
        is_wrapped=args.wrapped,
        temporal_coherence_file=args.temporal_coherence_file,
        temporal_coherence_threshold=args.temporal_coherence_threshold,
        conncomp_files=args.conncomp_files if not args.wrapped else None,
        exclude_zero_conncomps=args.exclude_zero_conncomps,
    )

    logger.info("Comparison completed. Output files:")
    for file in output_files:
        logger.info(f"  {file}")


if __name__ == "__main__":
    main()
