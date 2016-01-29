# TODO 
# put the selection option back in! 
# remove bokeh symbol and the link-to-this link? 
# Compare multiple portfolios?  Dropdown select? 

## HISTOGRAM BINS ON KDE ESTIMATE 
## FIX TITLE SIZES 
## Put times on x-axis
## Fix max number of plots in series 
## One of the tickers appears not to be overwritten for either drift, vol or derivatives, 
## test black sholes and note value is decreasing 
## include a calculate button instead of so many updates
## Make sure that we are getting MC error correct!!

## Questions: 
# how to control spacing between vtable and htable boxes? 

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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns

from bokeh.models import ColumnDataSource, Plot
from bokeh.plotting import figure, curdoc
from bokeh.properties import String, Instance
from bokeh.server.app import bokeh_app
from bokeh.server.utils.plugins import object_page
from bokeh.models.widgets import HBox, VBox, VBoxForm, Select, TextInput

from bokeh import mpl

TITLE_SIZE = 14

def payoff(x,k):
    return [max(val,0) for val in (x - k)]

class StockApp(VBox):
    extra_generated_classes = [["StockApp", "StockApp", "VBox"]]
    jsmodel = "VBox"

    Y = Instance(ColumnDataSource)

    # plots
    hist1 = Instance(Plot)
    hist2 = Instance(Plot)
    hist3 = Instance(Plot)

    # data source
    source = Instance(ColumnDataSource)
    risk_source = Instance(ColumnDataSource)

    # layout boxes
    mainrow = Instance(HBox)
    ticker1_box = Instance(HBox)
    ticker2_box = Instance(HBox)
    ticker3_box = Instance(HBox)
    ticker4_box = Instance(HBox)
    ticker5_box = Instance(HBox)
    second_row = Instance(HBox)
    histrow = Instance(HBox)

    # inputs
    ticker1  = String(default="1.2*(1.1-x)")
    ticker1p = String(default="-1.2")
    ticker2 = String(default="4.0")
    ticker2p = String(default="0.0")
    ticker3 = String(default="500")
    ticker3_1 = String(default="252")
    ticker3_2 = String(default="0.01")
    ticker4 = String(default="100")
    ticker4_1 = String(default="1.01")
    ticker4_2 = String(default="Milstein")
    button = String(default="")
    ticker1_select = Instance(TextInput)
    ticker1p_select = Instance(TextInput)
    ticker2_select = Instance(TextInput)
    ticker2p_select = Instance(TextInput)
    ticker3_select = Instance(TextInput)
    ticker3_1_select = Instance(TextInput)
    ticker3_2_select = Instance(TextInput)
    ticker4_select = Instance(TextInput)
    ticker4_1_select = Instance(TextInput)
    ticker4_2_select = Instance(Select)
    button_select = Instance(TextInput)
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
        obj.ticker1_box = HBox(width=500)
        obj.ticker2_box = HBox(width=500)
        obj.ticker3_box = HBox(width=467)
        obj.ticker4_box = HBox(width=500)
        obj.ticker5_box = HBox(width=500)
        obj.second_row = HBox()
        obj.histrow = HBox()
        obj.input_box = VBoxForm(width=600)

        # create input widgets
        obj.make_inputs()

        # outputs
        #obj.make_source()
        obj.main_mc(252,500,0.01,'Milstein',1.01)
        obj.make_plots()

        # layout
        obj.set_children()
        return obj

    def make_inputs(self):

        self.ticker1_select = TextInput(
            name='ticker1',
            title='Drift Function:',
            value='1.2*(1.1-x)',
        )
        self.ticker1p_select = TextInput(
            name='ticker1p',
            title='Drift Derivative:',
            value='-1.2',
        )
        self.ticker2_select = TextInput(
            name='ticker2',
            title='Volatility Function:',
            value='4.0',
        )
        self.ticker2p_select = TextInput(
            name='ticker2p',
            title='Volatility Derivative:',
            value='0.0',
        )
        self.ticker3_select = TextInput(
            name='ticker3',
            title='Number of Paths:',
            value='500'
        )
        self.ticker3_1_select = TextInput(
            name='ticker3_1',
            title='Number of Points:',
            value='252'
        )
        self.ticker3_2_select = TextInput(
            name='ticker3_2',
            title='Time Step:',
            value='0.01'
        )
        self.ticker4_select = TextInput(
            name='ticker4',
            title='Histogram Line:',
            value='100'
        )
        self.ticker4_1_select = TextInput(
            name='ticker4_1',
            title='Initial Value:',
            value='1.01'
        )
        self.ticker4_2_select = Select(
            name='ticker4_2',
            title='MC Scheme:',
            value='Milstein',
            options=['Euler','Milstein', 'Pred/Corr']
        )
        self.button_select = TextInput(
            name='button',
            title='Type any word containing "run" to run Simulation ',
            value = ''
        )

    
    def make_source(self):
        self.source = ColumnDataSource(data=self.Y)

    def main_mc(self,num_pts,num_paths, delta_t, method, Y0):
        def a(x):
            return eval(self.ticker1)
        def ap(x):
            return eval(self.ticker1p)
        def b(x): 
            return eval(self.ticker2)
        def bp(x):
            return eval(self.ticker2p)

        rpaths = np.random.normal(0, delta_t, size=(num_pts,num_paths))
        Y = np.array([[Y0]*num_paths]) 
        dt_vec = np.array([delta_t]*num_paths)

        if method == 'Milstein':
            for i in xrange(0,num_pts):
                tY = Y[-1,:]
                dW = rpaths[i,:]
                Y = np.vstack([Y, tY + a(tY)*dt_vec + b(tY)*dW + 0.5*b(tY)*bp(tY)*(dW*dW-dt_vec)])

        elif method == 'Pred/Corr':
            # Predictor corrector method is taken from equation 2.6 in this paper:
            # http://www.qfrc.uts.edu.au/research/research_papers/rp222.pdf
            rpaths2 = np.random.normal(0, delta_t, size=(num_pts,num_paths))

            for i in xrange(0,num_pts):
                tY = Y[-1,:]
                Ybar = tY + a(tY)*dt_vec + b(tY)*rpaths[i,:]
                abar_before = a(tY) - 0.5*b(tY)*bp(tY)  
                abar_after = a(Ybar) - 0.5*b(Ybar)*bp(Ybar)  
                Y = np.vstack([Y, tY + 0.5*(abar_before + abar_after)*dt_vec + 0.5*(b(tY)+b(Ybar))*rpaths2[i,:]])

        else:  # default to Euler Scheme 
            for i in xrange(0,num_pts):
                tY = Y[-1,:]
                Y = np.vstack([Y, tY + a(tY)*dt_vec + b(tY)*rpaths[i,:]])
    
        return Y  # return simulated paths 
    
    def path_plot(self):
        num_paths_plot = min(50,int(self.ticker3))
        hist_point = int(self.ticker4)
        #print 'Hist Point ', hist_point
        Y = self.Y.as_matrix()
        pltdat = Y[:,0:num_paths_plot]
        mY, MY = min(Y[hist_point,:]), max(Y[hist_point,:])

        plt.plot(pltdat, alpha=0.1, linewidth=1.8)
        sns.tsplot(pltdat.T,err_style='ci_band', ci=[68,95,99], alpha=1, \
                linewidth = 2.5, color='indianred')
        #sns.tsplot(pltdat.T,err_style='ci_band', ci=[68,95,99,99.99999], alpha=1, \
        #        linewidth = 2.5, condition='Mean Path', color='indianred')
        plt.plot([hist_point, hist_point], [0.99*mY, 1.01*MY], 'k-',label='Time Series Histogram')
        plt.xlabel('Time Step')
        plt.ylabel('Price')
        #plt.legend()

        p = mpl.to_bokeh()
        p.title = 'Mean Path (Red), MC Paths (Background) and Density Line (Black)'
        p.title_text_font_size= str(TITLE_SIZE)+'pt'

        return p

    def hist_den_plot(self):
        Y = self.Y.as_matrix()
        hist_point = int(self.ticker4)
        delta_t = float(self.ticker3_2)

        data = Y[hist_point,:]
        sns.distplot(data, color='k', hist_kws={"color":"b"}, norm_hist=True)
        #sns.distplot(data, color='k', hist_kws={"color":"b"})
        plt.hist(data)
        plt.title('Distribution at T = ' + str(np.round(delta_t*hist_point,4)) + ' with Mean: ' +str(np.round(np.mean(data),4)) + ' and Std Dev: ' + str(np.round(np.std(data),4)))
        plt.xlabel('Price Bins')
        plt.ylabel('Bin Count')
       
        p = mpl.to_bokeh()
        p.title_text_font_size= str(TITLE_SIZE)+'pt'

        return p

    def mc_results(self):
        # Compute Monte Carlo results 
        Y = self.Y.as_matrix()
        Y0 = float(self.ticker4_1)
        hist_point = int(self.ticker4)
        num_paths = int(self.ticker3)

        center_point = np.mean(Y[hist_point,:])
        stkgrid = np.linspace(0.5*center_point,1.5*center_point,100)
        meanlst = np.array([])
        stdlst  = np.array([])
        paylst  = np.array([])

        for stk in stkgrid:
            meanlst = np.append(meanlst, np.mean(payoff(Y[hist_point,:],stk)))
            stdlst = np.append(stdlst,np.std(payoff(Y[hist_point,:],stk))/np.sqrt(num_paths))

        plt.plot(stkgrid,meanlst+2*stdlst, 'g-')
        plt.plot(stkgrid,meanlst-2*stdlst,'g-',label='2-Sig Error')
        plt.plot(stkgrid,meanlst+stdlst,'r-')
        plt.plot(stkgrid,meanlst-stdlst,'r-',label='1-Sig Error')
        plt.plot(stkgrid,meanlst,'b',label='Mean')
        plt.title('MC Option Price (Blue) with 1-Sig (Red) and 2-Sig (Green) Errors')
        plt.xlabel('Strike')
        plt.ylabel('Value')

        p = mpl.to_bokeh()
        p.title_text_font_size= str(TITLE_SIZE)+'pt'

        return p

    def hist_plot(self):
        histdf = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
        #pltdf = self.source.to_df().set_index('date').dropna()
        #qlow, qhigh = mquantiles(pltdf[ticker],prob=[0.01,0.99]) 
        #tdf = pltdf[ticker]
        #histdf = tdf[((tdf > qlow) & (tdf < qhigh))]
        hist, bins = np.histogram(histdf, bins=50)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        start = bins.min()
        end = bins.max()
        top = hist.max()

        p = figure(
            title=' Histogram',
            plot_width=600, plot_height=400,
            tools="",
            title_text_font_size="16pt",
            x_range=[start, end],
            y_range=[0, top],
            x_axis_label = ' Bins',
            y_axis_label = 'Bin Count' 
        )
        p.rect(center, hist / 2.0, width, hist)
        return p

    def make_plots(self):

        self.hist_plots()

    def hist_plots(self):
        self.hist1 = self.path_plot()
        self.hist2 = self.hist_den_plot()
        self.hist3 = self.mc_results()

    def set_children(self):
        self.children = [self.mainrow, self.second_row]
        self.mainrow.children = [self.input_box, self.hist1]
        self.second_row.children = [self.hist2, self.hist3]
        self.input_box.children = [self.ticker1_box, self.ticker2_box, self.ticker3_box,self.ticker4_box,self.ticker5_box]
        self.ticker1_box.children =[self.ticker1_select, self.ticker1p_select]
        self.ticker2_box.children =[self.ticker2_select, self.ticker2p_select]
        self.ticker3_box.children =[self.ticker3_select, self.ticker3_1_select, self.ticker3_2_select]
        self.ticker4_box.children =[self.ticker4_select, self.ticker4_1_select, self.ticker4_2_select]
        self.ticker5_box.children =[self.button_select]

    def input_change(self, obj, attrname, old, new):
        if obj == self.ticker4_2_select:
            self.ticker4_2 = new
        if obj == self.ticker4_1_select:
            self.ticker4_1 = new
        if obj == self.ticker4_select:
            self.ticker4 = new
        if obj == self.ticker3_2_select:
            self.ticker3_2 = new
        if obj == self.ticker3_1_select:
            self.ticker3_1 = new
        if obj == self.ticker3_select:
            self.ticker3 = new
        if obj == self.ticker2p_select:
            self.ticker2p = new
        if obj == self.ticker2_select:
            self.ticker2 = new
        if obj == self.ticker1p_select:
            self.ticker1p = new
        if obj == self.ticker1_select:
            self.ticker1 = new
        if obj == self.button_select:
            self.button = new 
            if 'run' in self.button:
                self.make_source()
                self.make_plots()
                self.set_children()
                curdoc().add(self)

        #self.make_source()
        #self.make_plots()
        #self.set_children()
        #curdoc().add(self)

    def setup_events(self):
        super(StockApp, self).setup_events()
        if self.ticker1_select:
            self.ticker1_select.on_change('value', self, 'input_change')
        if self.ticker1p_select:
            self.ticker1p_select.on_change('value', self, 'input_change')
        if self.ticker2_select:
            self.ticker2_select.on_change('value', self, 'input_change')
        if self.ticker2p_select:
            self.ticker2p_select.on_change('value', self, 'input_change')
        if self.ticker3_select:
            self.ticker3_select.on_change('value', self, 'input_change')
        if self.ticker3_1_select:
            self.ticker3_1_select.on_change('value', self, 'input_change')
        if self.ticker3_2_select:
            self.ticker3_2_select.on_change('value', self, 'input_change')
        if self.ticker4_select:
            self.ticker4_select.on_change('value', self, 'input_change')
        if self.ticker4_1_select:
            self.ticker4_1_select.on_change('value', self, 'input_change')
        if self.ticker4_2_select:
            self.ticker4_2_select.on_change('value', self, 'input_change')
        if self.button_select:
            self.button_select.on_change('value',self, 'input_change')

    @property
    def Y(self):
        tmpdf = pd.DataFrame(self.main_mc(int(self.ticker3_1),int(self.ticker3),float(self.ticker3_2),self.ticker4_2,float(self.ticker4_1)))
        tmpdf.columns = ['Col_' + str(i) for i in xrange(len(tmpdf.columns))]
        #print tmpdf
        return tmpdf

# The following code adds a "/bokeh/stocks/" url to the bokeh-server. This URL
# will render this StockApp. If you don't want serve this applet from a Bokeh
# server (for instance if you are embedding in a separate Flask application),
# then just remove this block of code.
# http://localhost:5006/bokeh/stocks (URL for web app)
@bokeh_app.route("/bokeh/option_pricer/")
@object_page("option_pricer")
def make_option_pricer():
    app = StockApp.create()
    return app

