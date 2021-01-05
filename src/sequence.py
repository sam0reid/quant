
import h5py
import numpy as np
import os
import pandas as pd
import pandas_market_calendars as mcal
import pathlib
import datetime

from concurrent.futures import ProcessPoolExecutor

def hdf5ToDf(dset):
    """
    load hdf5 file to dataframe
    """
    
    sec_data =  h5py.File(dset, 'r')
    df = pd.DataFrame(data=sec_data['ohlcv'], 
                      index=sec_data['times'], 
                      columns=['open', 'high', 'low', 'close', 'volume'])
    df.index = [idx[0].replace("'","").strip('b') for idx in df.index]
    try:
        df.index = [idx.decode('utf-8') for idx in df.index]
    except (UnicodeDecodeError, AttributeError):
        pass
    
    df.index = [datetime.datetime.strptime(idx, "%Y%m%d %H:%M:%S") for idx in df.index]
    unique_df = df[~df.index.duplicated(keep='first')]
    
    return unique_df

def generate():
    """
    """
    sequence_path = pathlib.Path('sequences') 

    dset_to_ticker = lambda x : x.split('/')[1].strip('.hdf5')
    full_size = 48E6

    dsets = []
    for f in os.listdir('data'):
        p = pathlib.Path('data') / f
        if p.stat().st_size > full_size:
            dsets.append(p)

    periods = [5,10,20,40,80,160]
    for dset in dsets:
        df = hdf5ToDf(dset)
        for period in periods:
            df[f'return {period}'] = (df['close'] - df['close'].shift(period))/df['close'].shift(period)
            df[f'return {period} shifted'] = df[f'return {period}'].shift(-period)
            df.dropna(inplace=True)

            y = df[f'return {period} shifted']
            x = df[f'return {period}']
            time_stamps = y.index

            folder_name = dset_to_ticker(str(dset))
            if not os.path.exists(sequence_path / folder_name):
                os.mkdir(sequence_path / folder_name)

            np.save(sequence_path / folder_name / f'y_{period}.npy', y.values)
            np.save(sequence_path / folder_name / f'x_{period}.npy', x.values)
            np.save(sequence_path / folder_name / f'time_stamps_{period}.npy', time_stamps)

if __name__ == "__main__":
    generate()