#!/bin/sh

ENV_DIR='.venv'

if [[ ! -d $ENV_DIR ]]
then
    echo "Installing required packages."
    python3 -m venv $ENV_DIR
    source $ENV_DIR/bin/activate
    pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
else
    echo "Activating environment."
    source $ENV_DIR/bin/activate
fi

export PYTHONPATH='.'
