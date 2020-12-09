# type: ignore
from lpf.simulation.noise_extractor import NoiseExtractor
import yaml
import sys

def get_config():

    cmd = sys.argv
    if len(cmd) != 2:
        raise ValueError(
            f"Please provide path to config file. Example: extract_noise.py /path/to/config.yml"
        )
    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)  # type: ignore
    return config


def main():
    config = get_config()
    ne = NoiseExtractor(config)
    ne.run()


if __name__ == "__main__":
    main()