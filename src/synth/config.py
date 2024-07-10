from datetime import datetime
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ._types import Bbox, RhoOption


class SimulationInputs(BaseSettings):
    """Create parameters describing simulation data to generate."""

    model_config = SettingsConfigDict(cli_parse_args=True, cli_prog_name="synth")

    output_dir: Path = Path()
    bounding_box: Bbox = Field(
        ..., description="(left, bottom, right, top) in EPSG:4326"
    )
    start_date: datetime = datetime(2020, 1, 1)
    dt: int = Field(12, ge=1, le=365, description="Time step [days]")
    num_dates: int = Field(20, ge=2, le=1000)
    res_y: float = Field(15, ge=1, le=1000, description="Y resolution [meters]")
    res_x: float = Field(15, ge=1, le=1000, description="X resolution [meters]")
    include_turbulence: bool = True
    max_turbulence_amplitude: float = 5
    include_deformation: bool = True
    max_defo_amplitude: float = 5
    include_ramps: bool = True
    max_ramp_amplitude: float = 1.0
    include_stratified: bool = False
    rho_transform: RhoOption = RhoOption.SHRUNK

    block_shape: tuple[int, int] = Field(
        (128, 128),
        description=(
            "Size of (rows, cols) to process at one time when generating covariance"
            " matrices/samples.  Must have enough memory for several matrices of shape"
            " `(*block_shape, inps.num_dates, inps.num_dates)`"
        ),
    )
