# type: ignore
from lpf.simulation.transient_simulator import TransientSimulator
import yaml
import sys
import numpy as np
import os
import matplotlib.pyplot as plt

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
    tsim = TransientSimulator(config)
    tsim.run()

    examples = np.load(os.path.join(config['simulation_output_folder'],'data.npy'))[:32]
    parameters = np.load(os.path.join(config['simulation_output_folder'],'parameters.npy'))[:32]

    for i in range(len(examples)):
        burst = examples[i]
        p = parameters[i]
        plt.imshow(burst + np.random.randn(*burst.shape))
        title_str = (
            f"DM: {p[0]:.2f}\n"
            f"Fluence: {p[1]:.2f}\n"
            f"Width: {p[2]:.2f}\n"
            f"Index: {p[3]:.2f}\n"
            f"Peak Value: {burst.max():.2f}\n"
        )
        plt.axis('off')
        plt.title(title_str)
        plt.tight_layout()
        plt.savefig(os.path.join(config['simulation_output_folder'], f'example_{i}.png'))


if __name__ == "__main__":
    main()