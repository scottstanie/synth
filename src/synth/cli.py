import sys
from pathlib import Path

from synth.config import SimulationInputs


def get_cli_args() -> SimulationInputs:
    """Create a new `SimulationInputs` object from CLI args."""
    return SimulationInputs()


def main():
    """Open the simulation parameters and generate data."""
    inputs = None
    if len(sys.argv) == 2 and sys.argv[1] != "--help":
        try:
            p = Path(sys.argv[1])
            if p.exists() and p.suffix == ".json":
                with open(p) as f:
                    inputs = SimulationInputs.model_validate_json(f.read())
        except Exception:
            pass

    if inputs is None:
        inputs = get_cli_args()
    from synth.core import create_simulation_data

    create_simulation_data(inputs)


if __name__ == "__main__":
    main()
