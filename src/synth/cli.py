from pathlib import Path

from synth.config import SimulationInputs

# def get_cli_args() -> SimulationInputs:
#     """Create a new `SimulationInputs` object from CLI args."""
#     return SimulationInputs()


# def main():
#     """Open the simulation parameters and generate data."""
#     inputs = None
#     if len(sys.argv) == 2 and sys.argv[1] != "--help":
#         p = Path(sys.argv[1])
#         if p.exists() and p.suffix == ".json":
#             with open(p) as f:
#                 # Erase the args passed in if this is a filename
#                 sys.argv = sys.argv[:1]
#                 inputs = SimulationInputs.model_validate_json(f.read())

#     if inputs is None:
#         inputs = get_cli_args()
#     from synth.core import create_simulation_data

#     create_simulation_data(inputs)


def run():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare phase files to synthetic deformation."
    )
    parser.add_argument(
        "-f",
        "--params-file",
        type=Path,
        default=Path("simulation_params.json"),
        help="Paths to simulation parameters JSON file",
    )
    args = parser.parse_args()

    from synth.core import create_simulation_data

    with open(args.params_file) as f:
        inputs = SimulationInputs.model_validate_json(f.read())
        create_simulation_data(inputs)


# if __name__ == "__main__":
#     main()
