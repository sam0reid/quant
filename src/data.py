
from abc import ABCMeta, abstractmethod

import datetime
import os
import numpy as np 
import pandas as pd
 
from event import MarketEvent


class DataHandler(object):
    """
    DataHandler is an abstract base class providing an interface for all subsequent (inherited) data handlers (both live and historic).
    The goal of a (derived) DataHandler object is to output a generated set of bars (OHLCVI) for each symbol requested.
    This will replicate how a live strategy would function as current
    market data would be sent "down the pipe". Thus a historic and live system will be treated identically by the rest of the backtesting suite.
    """
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def get_latest_bar(self, symbol):
        """
        Returns the last bar updated.
        """
        raise NotImplementedError("Should implement get_latest_bar()")

    @abstractmethod
    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars updated.
        """
        raise NotImplementedError("Should implement get_latest_bars()")
    
    @abstractmethod
    def get_latest_bar_datetime(self, symbol):
        """
        Returns a Python datetime object for the last bar. 
        """
        raise NotImplementedError("Should implement get_latest_bar_datetime()") 
    
    @abstractmethod
    def get_latest_bar_value(self, symbol, val_type):
        """
        Returns one of the Open, High, Low, Close, Volume or OI from the last bar.
        """
        raise NotImplementedError("Should implement get_latest_bar_value()")

    @abstractmethod
    def get_latest_bars_values(self, symbol, val_type, N=1):
        """
        Returns the last N bar values from the
        latest_symbol list, or N-k if less available.
        """
        raise NotImplementedError("Should implement get_latest_bars_values()")

    @abstractmethod
    def update_bars(self):
        """
        Pushes the latest bars to the bars_queue for each symbol in a tuple OHLCVI format: (datetime, open, high, low, close, volume, open interest).
        """
        raise NotImplementedError("Should implement update_bars()")

class HistoricHDF5DataHandler(DataHandler):
    """
    HistoricCSVDataHandler is designed to read HDF5 files for each requested symbol
    from disk and provide an interface to obtain the "latest" bar in a manner identical
    to a live trading interface.
    """

    def __init__(self, events, hdf5_dir, symbol_list): 
        """
        Initialises the historic data handler by requesting the location of the HDF5 files and a list of symbols.
        It will be assumed that all files are of the form ’symbol.hdf5’, where symbol is a string in the list.
        
        Parameters:
        -----------
            events - The Event Queue.
            csv_dir - Absolute directory path to the CSV files.
            symbol_list - A list of symbol strings.
        """
        self.events = events
        self.csv_dir = hdf5_dir
        self.symbol_list = symbol_list
        self.symbol_data = {} 
        self.latest_symbol_data = {}
        self.continue_backtest = True
        self._open_convert_hdf5_files()

    def _open_convert_hdf5_files(self):
        """
        Opens the hdf5 files from the data directory, converting
        them into pandas DataFrames within a symbol dictionary.
        For this handler it will be assumed that the data is taken from Yahoo. Thus its format will be respected.
        """
        comb_index = None
        for s in self.symbol_list:
            # Load the CSV file with no header information, indexed on date self.symbol_data[s] = pd.io.parsers.read_csv(
            os.path.join(self.csv_dir, ’%s.csv’ % s), header=0, index_col=0, parse_dates=True, names=[
                                                                                                        ’datetime’, ’open’, ’high’,
        ’low’, ’close’, ’volume’, ’adj_close’ ]
        ).sort()
        # Combine the index to pad forward values
        if comb_index is None:
        comb_index = self.symbol_data[s].index
        else: comb_index.union(self.symbol_data[s].index)
        # Set the latest symbol_data to None
        self.latest_symbol_data[s] = []
        # Reindex the dataframes
        for s in self.symbol_list:
        self.symbol_data[s] = self.symbol_data[s].\
        reindex(index=comb_index, method=’pad’).iterrows()