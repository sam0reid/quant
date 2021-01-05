
import h5py
import pandas as pd
import datetime

def processDf(dset):
    sec_data =  h5py.File(dset, 'r')
    df = pd.DataFrame(data=sec_data['ohlcv'], 
                      index=sec_data['times'], 
                      columns=['open', 'high', 'low', 'close', 'volume'])
    df.index = [idx[0] for idx in df.index]
    try:
        df.index = [idx.decode('utf-8') for idx in df.index]
    except (UnicodeDecodeError, AttributeError):
        pass
    df.index = [datetime.datetime.strptime(idx, "%Y%m%d %H:%M:%S") for idx in df.index]
    unique_df = df[~df.index.duplicated(keep='first')]
    return unique_df