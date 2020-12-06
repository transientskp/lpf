import os
from typing import Union

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# Typing
opt_str = Union[None, str]
opt_int = Union[None, int]