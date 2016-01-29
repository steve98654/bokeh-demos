# Let's write functions to make the Euler, Milstein, and other MC schemes that will be used 
# for our GUI display eventually 
#
# We assume that SDE's take the form 
#
# dX_t = a(X_t)dt + b(X_t)dW_t
#

## SET THE X-AXIS ACCORDING TO THE TIME STEP THAT WE ARE USING 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set(color_codes=True)

num_pts = 100 
num_paths = 500
delta_t = 0.1 
Y0 = 2
hist_point = 10 
num_paths_plot = 50

def step_fct(x):
    return 1*(x>0)

# drift term 
def a(x): 
    return 1*(1.4 - x)

# vol term 
def b(x): 
    return np.array([1.8]*len(x))

# derivative of b 
def bp(x): 
    return np.array([0.0]*len(x))

def payoff(x,k):
    return [max(val,0) for val in (x - k)]


def main_mc(num_pts,num_paths, delta_t, method, Y0):
    def a(x):
        return 1*(1.4 - x)
    def ap(x):
        return -1
    def b(x): 
        return 0.5
    def bp(x):
        return 0.0

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

# Run the paths 
Y = main_mc(100,100,0.1,'Pred/Corr',1.1)

### PUT EACH OF THESE PLOTS IN A SEPARATE FUNCTION. 

# Begin the main displays here 
TITLE_SIZE = 16
AXIS_SIZE = 14

def path_plot(num_paths_plot, Y, hist_point):

    plt.figure()
    plt.plot(Y[0:num_paths_plot,:], alpha=0.1, linewidth=1.8)
    p1 = sns.tsplot(Y[0:num_paths_plot,:].T,err_style='ci_band', ci=[68,95,99,99.99999], alpha=1, \
            linewidth = 2.5, condition='Mean Path', color='indianred')
    p2 = plt.axvline(hist_point, color='k',label='Time Series Histogram')
    plt.title('MC Paths, Mean Path', fontsize=TITLE_SIZE)
    plt.xlabel('Time Step', fontsize=AXIS_SIZE)
    plt.ylabel('Price', fontsize=AXIS_SIZE)
    plt.legend()

def hist_den_plot(Y, hist_point,delta_t):
    plt.figure()
    data = Y[hist_point,:]
    sns.distplot(data, color='k', hist_kws={"color":"b"})
    plt.title('Distribution at time ' + str(np.round(delta_t*hist_point,4)) + ' with Mean: ' + str(np.round(np.mean(data),4)) + \
            ' and Std Dev: ' + str(np.round(np.std(data),4)), fontsize=TITLE_SIZE)
    plt.xlabel('Price Bins', fontsize=AXIS_SIZE)
    plt.ylabel('Bin Count', fontsize=AXIS_SIZE)

def mc_results(Y0,Y,hist_point):
    # Compute Monte Carlo results 
    center_point = np.mean(Y[hist_point,:])
    stkgrid = np.linspace(0.5*center_point,1.5*center_point,100)
    meanlst = np.array([])
    stdlst  = np.array([])
    paylst  = np.array([])

    for stk in stkgrid:
        meanlst = np.append(meanlst, np.mean(payoff(Y[hist_point,:],stk)))
        stdlst = np.append(stdlst,np.std(payoff(Y[hist_point,:],stk)))

    plt.figure()
    plt.plot(stkgrid,meanlst,'b',label='Mean')
    plt.plot(stkgrid,meanlst+stdlst,'r-')
    plt.plot(stkgrid,meanlst-stdlst,'r-',label='1-Sig Error')
    plt.plot(stkgrid,meanlst+2*stdlst, 'g-')
    plt.plot(stkgrid,meanlst-2*stdlst,'g-',label='2-Sig Error')
    plt.title('Options Value with Errors', fontsize=TITLE_SIZE)
    plt.xlabel('Strike', fontsize=AXIS_SIZE)
    plt.ylabel('Value', fontsize=AXIS_SIZE)
    plt.legend()


