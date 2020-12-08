# type: ignore
from lpf.simulation.transient_simulator import TransientsSimulator
import yaml
import sys

def get_config():

    cmd = sys.argv
    if len(cmd) != 2:
        raise ValueError(
            f"Please provide path to config file. Example: transients.py /path/to/config.yml"
        )
    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)  # type: ignore
    return config


def main():
    config = get_config()
    tsim = TransientsSimulator(config)
    tsim.run()


if __name__ == "__main__":
    main()