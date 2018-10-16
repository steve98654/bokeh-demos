import pandas as pd
import tia.bbg.datamgr as dm

def data_download(fields,tickers,start_date,end_date,write_path=None):
    ''' 
    Field from FLDS which you would like to get data 
    for from each stock in the index.
    start_date: start date of data ... e.g. '10/1/2017'
    end_date  : end data of data ... e.g. '10/5/2017'
    '''
    mgr = dm.BbgDataManager()

    stocks = mgr[tickers]

    data = stocks.get_historical(fields, start=start,end=end)

    if write_path:
        data[tickers].to_csv(write_path)

    return data

if __name__ == "__main__": 
    fields = ["PX_LAST"]
    start = '03/01/2015'
    end = '09/30/2018'

    tickers = ["IWO US Equity","ARSQX US Equity","POLRX US Equity","CUSEX US Equity","TYG US Equity","OAKIX US Equity","ARSQX US Equity",
               "AEF US Equity","PDRDX US Equity","TYG US Equity","PDBAX US Equity","GIBIX US Equity","ANGIX US Equity","IWD US Equity",
               "IWN US Equity","WOOD US Equity"]

    data = data_download(fields,tickers,start,end,write_path=None)
    data.columns = data.columns.droplevel(1)
    data.columns = [name.split(' ')[0] for name in data.columns.values]
    data.rename(columns={"IWD":"SC (Proxy)", "IWN":"CHASCVA (Proxy)", "WOOD":"Timber (Proxy)"},inplace=True)
    data['Total'] = data.sum(1)
    print data.head()
    data.to_pickle('P:\\Desktop\\testdata.pkl')

    '''
    for ticker in tickers:              
        filename = "corpdata_" + ticker.split(" ")[0] + "_" + start.replace('/','-') + '_' + end.replace('/','-') + '.csv'
        write_path = 'C:\\Users\\smt\\Documents\\altdata\\' + filename
        data = data_download(fields,[ticker],start,end,write_path)
    '''
