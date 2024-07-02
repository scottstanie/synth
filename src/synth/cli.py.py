from troposim.ui import SimulationInputs, create_simulation_data


def main():
    with open("simulation_params.json") as f:
        inputs = SimulationInputs.model_validate_json(f.read())
    create_simulation_data(inputs)


if __name__ == "__main__":
    main()
