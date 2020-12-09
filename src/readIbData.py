
from pathlib import Path

import h5py
import numpy as np
import os 
import pandas as pd

if __name__ == "__main__":

    data_path = Path('data/MMM.hdf5')
    hdf5_output = h5py.File(data_path, 'r')

    for i, k in enumerate(hdf5_output.keys()):
        print(i)
        print(k)
        print(hdf5_output[k][:])