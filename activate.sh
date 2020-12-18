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

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PYTHONPATH=$DIR
export JUPYTER_PATH=$DIR
