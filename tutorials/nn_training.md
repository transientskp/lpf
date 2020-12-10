# Live Pulse Finder: Training your NN
This shows how to train a neural network (NN) for parameter inference from dynamic spectra data that the telescope you use can output. 
## 1. Simulate transients.
1. Create a parameter configuration file accustomed to your telescope (e.g., `examples/transients_a12.yml`). For a sparsely sampled frequency range you still need to specify the entire range and the delta-frequency. Later this will be processed further.
2. Run the `transients.py` script with as argument the path to your configuration file. E.g., `python lpf/simulation/scripts/transients.py examples/transients_a12.yml`
## 2. (Optional: extract noise from survey data) 
This extracts background noise for the dynamic spectra. If skipped, you'll use Gaussian noise.
1. Specify correct parameters in a YAML file. E.g., `examples/extract_noise.yml`. 
2. Run the noise extractor: 
`python lpf/simulation/scripts/extract_noise.py examples/extract_noise_a6.yml`

## 3. Neural Network training
1. Create a training file (for example `lpf/_nn/scripts/train_a12.py`) with the correct parameters and neural network architecture in a YAML file (e.g., `examples/nn_a12.yml`).
2. Run your training file, e.g. `python lpf/_nn/scripts/train_a12.py examples/nn_a12.yml`
3. Wait until it's converged!
