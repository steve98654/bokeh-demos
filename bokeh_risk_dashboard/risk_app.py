# TODO 
# put the selection option back in! 
# remove bokeh symbol and the link-to-this link? 
# Compare multiple portfolios?  Dropdown select? 

## Questions: 
# how to control spacing between vtable and htable boxes? 

### ADD A PERCENTAGE OF UP DAYS INTO THE DF 


# Done: 
# sourced with custom stock data 
# rearranged plots, made histogram 

"""
This file demonstrates a bokeh applet, which can either be viewed
directly on a bokeh-server, or embedded into a flask application.
See the README.md file in this directory for instructions on running.
"""

import logging

logging.basicConfig(level=logging.DEBUG)

from os import listdir
from os.path import dirname, join, splitext

import numpy as np
import math 
import pandas as pd
from scipy.stats.mstats import mquantiles 
from scipy.stats import skew 

from bokeh.models import ColumnDataSource, Plot
from bokeh.plotting import figure, curdoc
from bokeh.properties import String, Instance
from bokeh.server.app import bokeh_app
from bokeh.server.utils.plugins import object_page
from bokeh.models.widgets import HBox, VBox, VBoxForm, PreText, Select, TextInput, DataTable, TableColumn, NumberFormatter

# build up list of stock data in the daily folder
maindf = pd.read_pickle('open_price_new.pkl')
maindf.index = pd.to_datetime(maindf.index)
maindf.index = pd.to_datetime([val.date() for val in pd.to_datetime(maindf.index)])
maindf.index.name = 'date'

# Risk metric functions
# See wikipedia

def my_round(x,n):  
    if math.isnan(x):
        return -123
    elif x == 0:
        return 0.000
    elif x < 0: 
        nx = -x
        return -round(nx, -int(math.floor(math.log10(nx))) + (n - 1)) 
    else:
        return round(x, -int(math.floor(math.log10(x))) + (n - 1)) 

def my_skewness(srs): 
    return skew(srs)

def my_ex_kurtosis(srs): 
    n = float(len(srs))
    srsmean = np.mean(srs)
    g2 = np.mean((srs - srsmean)**4)/(np.mean((srs-srsmean)**2))**2 - 3
    return (n-1)/((n-2)*(n-3))*((n+1)*g2+6)

def draw_down(srs): 
    i = np.argmax(np.maximum.accumulate(srs) - srs) # end of the period
    j = np.argmax(srs[:i]) # start of period
    return srs[i] - srs[j]

def daily_returns_df(df): 
    return df/df.shift(1) - 1  

def daily_cum_returns_df(df,winsz): 
    return pd.rolling_sum(df,winsz).dropna()

def per_up_days_df(df,winsz):
    n = float(len(df))
    rtndf = daily_returns_df(df)
    return pd.rolling_apply(rtndf, winsz, lambda x: len([v for v in x if v > 0])/float(winsz))

def annualized_volatility_df(df,winsz): 
    return np.sqrt(252*pd.rolling_var(daily_returns_df(df),winsz))

def sharpe_ratio_df(df,winsz):
    rtndf = daily_returns_df(df)
    return np.sqrt(252)*pd.rolling_mean(rtndf,winsz)/np.sqrt(pd.rolling_var(rtndf,winsz))

def worst_daily_loss_df(df,winsz): 
    tmpdf = daily_returns_df(df)
    return pd.rolling_apply(tmpdf,winsz,lambda x: min(x))

def VaR_df(df,winsz,qtile): 
    tmpdf = daily_returns_df(df)
    return pd.rolling_quantile(tmpdf,winsz,qtile)  

def draw_down_df(df,winsz): 
    tmpdf = daily_returns_df(df).dropna().cumsum() 
    return pd.rolling_apply(tmpdf,winsz,lambda x: draw_down(x))

def get_ticker_data(ticker):
    data = maindf[ticker].dropna()
    data = pd.DataFrame({ticker: data.values},index=data.index)
    return data

def get_data(ticker1):
    return get_ticker_data(ticker1).dropna()

class StockApp(VBox):
    extra_generated_classes = [["StockApp", "StockApp", "VBox"]]
    jsmodel = "VBox"

    # text statistics
    pretext = Instance(DataTable)

    # plots
    line_plot1 = Instance(Plot)
    hist1 = Instance(Plot)

    # data source
    source = Instance(ColumnDataSource)
    risk_source = Instance(ColumnDataSource)

    # layout boxes
    mainrow = Instance(HBox)
    histrow = Instance(HBox)
    statsbox = Instance(VBox)

    # inputs
    ticker1 = String(default="INTC")
    ticker2 = String(default="Daily Prices")
    ticker3 = String(default='63')
    ticker4 = String(default='2010-01-01')
    ticker5 = String(default='2015-08-01')
    ticker1_select = Instance(Select)
    ticker2_select = Instance(Select)
    ticker3_select = Instance(TextInput)
    ticker4_select = Instance(TextInput)
    ticker5_select = Instance(TextInput)
    input_box = Instance(VBoxForm)

    def __init__(self, *args, **kwargs):
        super(StockApp, self).__init__(*args, **kwargs)
        self._dfs = {}

    @classmethod
    def create(cls):
        """
        This function is called once, and is responsible for
        creating all objects (plots, datasources, etc)
        """
        # create layout widgets
        obj = cls()
        obj.mainrow = HBox()
        obj.histrow = HBox()
        obj.statsbox = VBox()
        obj.input_box = VBoxForm()

        # create input widgets
        obj.make_inputs()

        # outputs
        #obj.pretext = PreText(text="", width=300)
        obj.pretext = DataTable(width=300)
        obj.make_source()
        obj.make_plots()
        obj.make_stats()

        # layout
        obj.set_children()
        return obj

    def make_inputs(self):

        self.ticker1_select = Select(
            name='ticker1',
            title='Portfolio:',
            value='MSFT',
            options = ['INTC', 'Tech Basket', 'IBB', 'IGOV']
        )
        self.ticker2_select = Select(
            name='ticker2',
            title='Risk/Performance Metric:',
            value='Price',
            options=['Daily Prices', 'Daily Returns', 'Daily Cum Returns', 'Max DD Percentage', 'Percentage Up Days', 'Rolling 95% VaR', 'Rolling Ann. Volatility', 'Rolling Worst Dly. Loss', 'Ann. Sharpe Ratio']
        )
        self.ticker3_select = TextInput(
            name='ticker3',
            title='Window Size:',
            value='63'
        )
        self.ticker4_select = TextInput(
            name='ticker4',
            title='Start Date:',
            value='2010-01-01'
        )
        self.ticker5_select = TextInput(
            name='ticker5',
            title='End Date:',
            value='2015-08-01'
        )

    @property
    def selected_df(self):
        pandas_df = self.df
        selected = self.source.selected['1d']['indices']
        if selected:
            pandas_df = pandas_df.iloc[selected, :]
        return pandas_df

    def make_source(self):
        if self.ticker2 == 'Daily Prices': 
            self.source = ColumnDataSource(data=self.df)
        elif self.ticker2 == 'Daily Returns': 
            self.source = ColumnDataSource(data=daily_returns_df(self.df))
        elif self.ticker2 == 'Daily Cum Returns': 
            self.source = ColumnDataSource(data=daily_cum_returns_df(daily_returns_df(self.df),int(self.ticker3)))
        elif self.ticker2 == 'Rolling Ann. Volatility': 
            self.source = ColumnDataSource(data=annualized_volatility_df(self.df,int(self.ticker3)))
        elif self.ticker2 == 'Rolling Worst Dly. Loss': 
            self.source = ColumnDataSource(data=worst_daily_loss_df(self.df,int(self.ticker3)))
        elif self.ticker2 == 'Rolling 95% VaR': 
            self.source = ColumnDataSource(data=VaR_df(self.df,int(self.ticker3),0.95))
        elif self.ticker2 == 'Ann. Sharpe Ratio':
            self.source = ColumnDataSource(data=sharpe_ratio_df(self.df,int(self.ticker3)))
        elif self.ticker2 == 'Max DD Percentage':
            self.source = ColumnDataSource(data=draw_down_df(self.df,int(self.ticker3)))
        elif self.ticker2 == 'Percentage Up Days':
            self.source = ColumnDataSource(data=per_up_days_df(self.df,int(self.ticker3)))
        else:
            self.source = ColumnDataSource(data=self.df)

    def line_plot(self, ticker, x_range=None):

        if self.ticker2 in ['Daily Prices', 'Daily Returns']:
            tltstr = self.ticker1 + ' ' + self.ticker2 
        else:
            tltstr=self.ticker1 + ' '  +self.ticker2 + ' with ' + self.ticker3 + ' Day Trailing Window'

        p = figure(
            title=tltstr,
            x_range=x_range,
            x_axis_type='datetime',
            plot_width=1200, plot_height=400,
            title_text_font_size="16pt",
            tools="pan,box_zoom,wheel_zoom,reset",
            x_axis_label = 'Date',
            y_axis_label = self.ticker2
        )

        p.line(
            'date', ticker,
            line_width=2,
            line_join='bevel',
            source=self.source,
            nonselection_alpha=0.02
        )
        return p

    def hist_plot(self, ticker):
        pltdf = self.source.to_df().set_index('date').dropna()
        qlow, qhigh = mquantiles(pltdf[ticker],prob=[0.01,0.99]) 
        tdf = pltdf[ticker]
        histdf = tdf[((tdf > qlow) & (tdf < qhigh))]
        hist, bins = np.histogram(histdf, bins=50)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        start = bins.min()
        end = bins.max()
        top = hist.max()

        p = figure(
            title=self.ticker1 + ' ' + self.ticker2 + ' Histogram',
            plot_width=600, plot_height=400,
            tools="",
            title_text_font_size="16pt",
            x_range=[start, end],
            y_range=[0, top],
            x_axis_label = self.ticker2 + ' Bins',
            y_axis_label = 'Bin Count' 
        )
        p.rect(center, hist / 2.0, width, hist)
        return p

    def make_plots(self):
        ticker1 = self.ticker1
        ticker2 = self.ticker2

        self.line_plot1 = self.line_plot(ticker1)
        self.hist_plots()

    def hist_plots(self):
        ticker1 = self.ticker1
        ticker2 = self.ticker2
        self.hist1 = self.hist_plot(ticker1)
        #self.hist2 = self.hist_plot(ticker2)

    def set_children(self):
        self.children = [self.mainrow, self.line_plot1]
        self.mainrow.children = [self.input_box, self.hist1, self.pretext]
        self.input_box.children = [self.ticker1_select, self.ticker2_select, self.ticker3_select, self.ticker4_select, self.ticker5_select]
        #self.statsbox.children = [self.pretext]

    def input_change(self, obj, attrname, old, new):
        if obj == self.ticker5_select:
            self.ticker5 = new
        if obj == self.ticker4_select:
            self.ticker4 = new
        if obj == self.ticker3_select:
            self.ticker3 = new
        if obj == self.ticker2_select:
            self.ticker2 = new
        if obj == self.ticker1_select:
            self.ticker1 = new

        self.make_source()
        self.make_stats()
        self.make_plots()
        self.set_children()
        curdoc().add(self)

    def setup_events(self):
        super(StockApp, self).setup_events()
        if self.source:
            self.source.on_change('selected', self, 'selection_change')
        if self.ticker1_select:
            self.ticker1_select.on_change('value', self, 'input_change')
        if self.ticker2_select:
            self.ticker2_select.on_change('value', self, 'input_change')
        if self.ticker3_select:
            self.ticker3_select.on_change('value', self, 'input_change')
        if self.ticker4_select:
            self.ticker4_select.on_change('value', self, 'input_change')
        if self.ticker5_select:
            self.ticker5_select.on_change('value', self, 'input_change')

    def make_stats(self):
        ## Build up a list of Summary stats for the time series 
        statsdf = self.source.to_df().dropna()
        stats = statsdf.describe()
        stats.index = ['Data Point Count', 'Mean', 'Standard Deviation', 'Minimum', '25%-ile',
        '50%-ile', '75%-ile', 'Maximum']
        stats.loc['Skewness',:] = my_round(my_skewness(statsdf[self.ticker1].values),4)
        stats.loc['Excess Kurtosis',:] = my_round(my_ex_kurtosis(statsdf[self.ticker1].values),4)
        stats = stats.fillna(-10)
        stats.index.name = "stats"
        columns = [TableColumn(field='stats', title='Time Series Statistic'),
                TableColumn(field=self.ticker1, title="Value",
                    formatter=NumberFormatter(format='0.0000'))]    
        self.pretext.columns = columns
        self.pretext.source = ColumnDataSource(data=stats)

    def selection_change(self, obj, attrname, old, new):
        self.make_stats()
        self.hist_plots()
        self.set_children()
        curdoc().add(self)

    @property
    def df(self):
        return get_data(self.ticker1).truncate(before=self.ticker4, after=self.ticker5)


# The following code adds a "/bokeh/stocks/" url to the bokeh-server. This URL
# will render this StockApp. If you don't want serve this applet from a Bokeh
# server (for instance if you are embedding in a separate Flask application),
# then just remove this block of code.
# http://localhost:5006/bokeh/stocks (URL for web app)
@bokeh_app.route("/bokeh/stocks/")
@object_page("stocks")
def make_stocks():
    app = StockApp.create()
    return app

