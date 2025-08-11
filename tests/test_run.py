from pathlib import Path

from synth.config import SimulationInputs
from synth.core import create_simulation_data

DATA_FILE = Path(__file__).parent / "sample.json"


def test_run():
    with open(DATA_FILE) as f:
        inputs = SimulationInputs.model_validate_json(f.read())
        create_simulation_data(inputs)
