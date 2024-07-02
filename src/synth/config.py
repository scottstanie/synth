from datetime import datetime
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ._types import Bbox


class SimulationInputs(BaseSettings):
    """Create parameters describing simulation data to generate."""

    model_config = SettingsConfigDict(cli_parse_args=True)

    output_dir: Path = Path()
    start_date: datetime
    dt: int = Field(..., ge=1, le=365, description="Time step [days]")
    num_dates: int = Field(..., ge=2, le=100)
    res_y: float = Field(..., ge=1, le=1000, description="Y resolution [meters]")
    res_x: float = Field(..., ge=1, le=1000, description="X resolution [meters]")
    bounding_box: Bbox = Field(
        ..., description="(left, bottom, right, top) in EPSG:4326"
    )
    include_turbulence: bool = True
    max_turbulence_amplitude: float = 5
    include_deformation: bool = True
    max_defo_amplitude: float = 5
    include_ramps: bool = True
    max_ramp_amplitude: float = 1.0
    include_stratified: bool = False
