# Life Pulse Finder: Tutorial
1. See for example `examples/aartfaac12.yml`. Specify the correct parameters. If you don't have a trained neural network yet, see `nn_training.md`.
2. Run `lpf examples/aartfaac12.yml`
3. The parameters of analyzed transients will be output to a `.csv` in the specified output folder. This can be opened for analysis. The `.npy` file in the output folder constains all the time-frequency datapoints. Index this at the correct source ID and time-step (specified in the `.csv` file to manually inspect the data.