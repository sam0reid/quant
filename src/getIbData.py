

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import BarData

from pathlib import Path
from threading import Thread

import datetime
import h5py
import os
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import queue
import time
import tqdm

MAX_WAIT_SECONDS = 10
HISTORICAL_DATA_TIMEOUT = object()
    
class FileAlreadyExists(Exception):
    pass

class MessageProcessor(object):
    """
    Store messages as they come in to hdf5 filee
    """
    def __init__(self, data_path:Path, size:int=1000000):
        self.message_queue = queue.Queue()
        self.size = size
        self.data_path = data_path

        hdf5_output = h5py.File(data_path, 'w')
        hdf5_output.close()

    def push(self, df:pd.DataFrame):
        """
        Push mesages onto queue unless its full, in that case messages are flushed

        Parameters
        ----------
            message
        """
        if self.message_queue.qsize() < self.size:
            self.message_queue.put(df)
        else:
            self.flush()
            self.message_queue.put(df)
    
    def flush(self):
        """
        Flush the queue by storing data. Assumes messages are bars structured
        as open, high, low, close. Stores time separately as strings
        """

        hdf5_output = h5py.File(self.data_path, 'r+')
        ohlcv = pd.concat(self.message_queue.queue)
        ohlcv_arr = ohlcv.to_numpy()
        times_arr = ohlcv.index.values.reshape((-1,1))
        string_dt = h5py.special_dtype(vlen=str)
    
        if 'ohlcv' not in hdf5_output.keys():
            hdf5_output.create_dataset('ohlcv',
                                        maxshape=(None,ohlcv_arr.shape[1]), 
                                        data=ohlcv_arr)
            hdf5_output.create_dataset('times',
                                maxshape=(None,1), 
                                data=times_arr,
                                dtype=string_dt)
        else:
            ohlcv_dataset = hdf5_output['ohlcv']
            times_dataset = hdf5_output['times']

            ohlcv_dataset.resize(ohlcv_dataset.shape[0] + ohlcv_arr.shape[0], axis=0)
            ohlcv_dataset[-ohlcv_arr.shape[0]:,:] = ohlcv_arr
            
            times_dataset.resize(times_dataset.shape[0] + times_arr.shape[0], axis=0)
            times_dataset[-times_arr.shape[0]:,:] = times_arr
        hdf5_output.close()
        self.message_queue = queue.Queue()

class MyWrapper(EWrapper):

    def __init__(self):
        self._my_resolved_contracts = {}
        self.finished_requests = queue.Queue()

    def initcontractdetails(self, reqId:int):
        self.contract_details_queue = queue.Queue()
        return self.contract_details_queue

    def initresolvedcontract(self, reqId:int):
        self._my_resolved_contracts[reqId] = queue.Queue()
        return self._my_resolved_contracts[reqId]

    def initmessageprocessor(self, name:str, reqId:int):
        self._my_message_processor = MessageProcessor(name)
        return self._my_message_processor

    def nextValidId(self, orderId:int):
        # overwritten
        print("setting nextValidOrderId: %d", orderId)
        self.nextValidOrderId = orderId

    def historicalData(self, reqId:int, bar: BarData):
        # overwritten
        self._my_message_processor.push(barToDataframe(bar))

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        # overwritten
        self._my_message_processor.flush()
        self.finished_requests.put(reqId)

    def error(self, reqId, errorCode, errorString):
        # overwritten
        print("Error. Id: " , reqId, " Code: " , errorCode , " Msg: " , errorString)

    def symbolSamples(self, reqId: int, contractDescriptions):
        super().symbolSamples(reqId, contractDescriptions)
        self.contract_details_queue.put(contractDescriptions)

    def contractDetails(self, reqId: int, contractDetails):
        super().contractDetails(reqId, contractDetails)
        self._my_resolved_contracts[reqId].put(contractDetails)


class MyClient(EClient):
    def __init__(self, wrapper):
        EClient.__init__(self, wrapper)

    def getMostLikelyContract(self, ticker, reqId:int):
        """
        Turn ticker into real contract
        """
        contract_details_queue = self.wrapper.initcontractdetails(reqId)
        print("Getting full contract details from the server... ")
        self.reqMatchingSymbols(reqId, ticker)
        ## Run until we get a valid contract(s) or get bored waiting
        contract_details = contract_details_queue.get(timeout = MAX_WAIT_SECONDS)
        first_contract = contract_details[0].contract

        resolved_contract_queue = self.wrapper.initresolvedcontract(reqId)
        self.reqContractDetails(reqId, first_contract)
        resolved_contract = resolved_contract_queue.get(timeout = MAX_WAIT_SECONDS)
        return resolved_contract.contract

    def initializeMessageProcessor(self, contract, reqId:int):
        fname = Path('data') / f"{contract.symbol}.hdf5"
        self.wrapper.initmessageprocessor(fname, reqId)

    def getHistoricalData(self, contract:Contract, end:int, duration:str, interval:str, reqId:int):
        """
        Get historical data for given ticker. Waits for response from wrapper before
        completing
        
        Parameters
        ----------
            contract: IB contract
            end: the number of days, before today that marks the end of the query period
            duration: string like '1 D' '1 Y' etc. (see IB docs)
            interval: string like '1 min' '8 hours' etc. (see IB docs)
            reqId: integer unique to request at hand

        Returns
        -------
            integer value if completed successfully
            a locally defined object if timeout or other error occured.
        """
        self.reqHistoricalData(reqId, contract, end, duration, interval, "MIDPOINT", 1, 1, False, [])
        try:
            finished = self.wrapper.finished_requests.get(timeout=MAX_WAIT_SECONDS*100)
            return finished
        except:
            return HISTORICAL_DATA_TIMEOUT

class MyApp(MyWrapper, MyClient):
    def __init__(self, ipaddress, portid, clientid):
        MyWrapper.__init__(self)
        MyClient.__init__(self, wrapper=self)

        self.connect(ipaddress, portid, clientid)

        thread = Thread(target = self.run)
        thread.start()

        setattr(self, "_thread", thread)

def barToDataframe(bar: BarData) -> pd.DataFrame():
    """
    Form dataframe from bar
    
    Parameters
    ----------
        bar object from IB

    Returns
    -------
        pandas dataframe
    """
    return pd.DataFrame({'open' : bar.open, 
                        'high' : bar.high, 
                        'low' : bar.low, 
                        'close' : bar.close, 
                        'volume' : bar.volume}, 
                        index=[np.string_(bar.date)])

def getSPYTickers():
    payload=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = payload[0]
    symbols = df['Symbol'].values.tolist()
    return symbols

if __name__ == "__main__":

    app = MyApp("127.0.0.1", 7497, 1)
    symbols = getSPYTickers()
    failed = False
    num_years = 5
    for i, s in enumerate(symbols):
        con = app.getMostLikelyContract(s, i)
        data_path = Path(r'data') / f"{con.symbol}.hdf5"
        if os.path.exists(data_path):
            pass
        else:
            nyse = mcal.get_calendar('NYSE')

            nyse_schedule = nyse.schedule(start_date=datetime.datetime.today() - datetime.timedelta(days=365*num_years), 
                                          end_date=datetime.datetime.today())
            t_steps = [ts.strftime("%Y%m%d %H:%M:%S") for ts in nyse_schedule.index]
            j=0
            pbar = tqdm.tqdm(total=len(t_steps))
            pbar.set_description(f"Getting data for {con.symbol}")
            app.initializeMessageProcessor(con, i)
            while j < len(t_steps):
                pbar.update(1)
                result = app.getHistoricalData(con, t_steps[j], "1 D", "1 min", int(f"{i}"))
                if result == HISTORICAL_DATA_TIMEOUT:
                    app.disconnect()
                    failed = True
                    break
                else:
                    j+=1
            if failed:
                break
            pbar.close()
    app.disconnect()