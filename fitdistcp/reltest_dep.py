import numpy as np
import json
import warnings
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import expon
from scipy.stats import genextreme
from scipy.stats import genpareto
from scipy.stats import gumbel_r

import gumbel as cp_gumbel
import gumbel_libs as cp_gumbel_libs
import expon as cp_expon
import expon_libs as cp_expon_libs
import genpareto as cp_genpareto
import genpareto_libs as cp_genpareto_libs
import genextreme as cp_genextreme
import genextreme_libs as cp_genextreme_libs

import reltest_libs


warnings.filterwarnings("ignore", category=RuntimeWarning)


def reltest_expon(ntrials=100, nx=30, p=0.0001*np.asarray(range(1,10000)), scale=1):
    '''
    Reliability test.

    Parameters
    ----------
    desired_p : array_like
        Probabilities at which to calculate quantiles.
    ntrials : int
        Number of trials to average over.
    nx : int
        Number of samples per trial.
    scale: float (default = 1)
        Scale parameter.
    
    Returns
    -------
    dict
        Dictionary with keys:
            'actual_p_ml' : array_like
                Achieved probabilities using ML quantiles.
            'actual_p_cp' : array_like
                Achieved probabilities using CP quantiles.

    Each trial generates nx samples and calculates quantiles using the two methods.
    The difference between the methods is clearest when nx is in the range of 20-60.
    Increasing ntrials reduces the effect of random variations in the trials (100 is sufficient for many purposes).
    '''

    p_actual_ml_total = np.zeros(len(p))
    p_actual_cp_total = np.zeros(len(p))

    for i in range(ntrials):
        x = expon.rvs(scale=scale, size=nx)
        
        # calculate quantile using ML and CP methods
        # ML:
        ics = [scale]
        scale_ml = minimize(lambda params: -cp_expon_libs.exp_logf(params, x), ics, method='BFGS').x 
        q_ml = expon.ppf(p, scale=scale_ml)

        # CP:
        q_cp = cp_expon.ppf(x, p)['cp_quantiles']

        # feed back in for the actual probability
        p_actual_ml_total += expon.cdf(q_ml, scale=scale)
        p_actual_cp_total += expon.cdf(q_cp, scale=scale)

    p_actual_ml_avg = p_actual_ml_total / ntrials
    p_actual_cp_avg = p_actual_cp_total / ntrials

    result = {
        'actual_p_ml' : np.ndarray.tolist(p_actual_ml_avg), 
        'actual_p_cp': np.ndarray.tolist(p_actual_cp_avg), 
        'p': np.ndarray.tolist(p)
        }

    return result




def reltest_genextreme(ntrials=100, nx=30, p=0.0001*np.asarray(range(1,10000)), xi=0, loc=0, scale=1):
    '''
    Reliability test.

    Parameters
    ----------
    desired_p : array_like
        Probabilities at which to calculate quantiles.
    ntrials : int
        Number of trials to average over.
    nx : int
        Number of samples per trial.
    xi: float (default = 0)
        Shape parameter to test.
    loc: float (default = 0)
        Loc parameter to test.
    scale: float (default = 1)
        Scale parameter to test.
    
    Returns
    -------
    dict
        Dictionary with keys:
            'actual_p_ml' : array_like
                Achieved probabilities using ML quantiles.
            'actual_p_cp' : array_like
                Achieved probabilities using CP quantiles.

    Each trial generates nx samples and calculates quantiles using the two methods.
    The difference between the methods is clearest when nx is in the range of 20-60.
    Increasing ntrials reduces the effect of random variations in the trials (100 is sufficient for many purposes).
    '''

    p_actual_ml_total = np.zeros(len(p))
    p_actual_cp_total = np.zeros(len(p))

    for i in range(ntrials):
        x = genextreme.rvs(-xi, loc=loc, scale=scale, size=nx)
        
        # calculate quantile using ML and CP methods
        # ML:
        ics = [loc, scale, xi]
        mu_ml, sigma_ml, xi_ml = minimize(lambda params: -cp_genextreme_libs.gev_loglik(params, x), ics, method='BFGS').x
        q_ml = genextreme.ppf(p, -xi_ml, loc=mu_ml, scale=sigma_ml) 

        # CP:
        q_cp = cp_genextreme.ppf(x, p)['cp_quantiles']   

        # feed back in for the actual probability
        p_actual_ml_total += genextreme.cdf(q_ml, -xi, loc=loc, scale=scale)
        p_actual_cp_total += genextreme.cdf(q_cp, -xi, loc=loc, scale=scale)

    p_actual_ml_avg = p_actual_ml_total / ntrials
    p_actual_cp_avg = p_actual_cp_total / ntrials

    result = {
        'actual_p_ml' : np.ndarray.tolist(p_actual_ml_avg), 
        'actual_p_cp': np.ndarray.tolist(p_actual_cp_avg), 
        'p': np.ndarray.tolist(p)
        }

    return result


def reltest_genpareto(ntrials=100, nx=30, p=0.0001*np.asarray(range(1,10000)), kloc=0, scale=1, xi=0):
    '''
    Reliability test.

    Parameters
    ----------
    desired_p : array_like
        Probabilities at which to calculate quantiles.
    ntrials : int
        Number of trials to average over.
    nx : int
        Number of samples per trial.
    kloc: float (default = 0)
        Loc parameter to test.
    xi: float (default = 0)
        Shape parameter to test.
    scale: float (default = 1)
        Scale parameter to test.
    
    Returns
    -------
    dict
        Dictionary with keys:
            'actual_p_ml' : array_like
                Achieved probabilities using ML quantiles.
            'actual_p_cp' : array_like
                Achieved probabilities using CP quantiles.

    Each trial generates nx samples and calculates quantiles using the two methods.
    The difference between the methods is clearest when nx is in the range of 20-60.
    Increasing ntrials reduces the effect of random variations in the trials (100 is sufficient for many purposes).
    '''

    p_actual_ml_total = np.zeros(len(p))
    p_actual_cp_total = np.zeros(len(p))

    for i in range(ntrials):
        x = genpareto.rvs(-xi, loc=kloc, scale=scale, size=nx)
        
        # calculate quantile using ML and CP methods
        # ML:
        ics = [scale, xi]
        sigma_ml, xi_ml = minimize(lambda params: -cp_genpareto_libs.gpd_k1_loglik(params, x, kloc), ics, method='BFGS').x
        q_ml = genpareto.ppf(p, -xi_ml, loc=kloc, scale=sigma_ml)

        # CP:
        q_cp = cp_genpareto.ppf(x, p)['cp_quantiles']

        # feed back in for the actual probability
        p_actual_ml_total += genpareto.cdf(q_ml, -xi, loc=kloc, scale=scale)
        p_actual_cp_total += genpareto.cdf(q_cp, -xi, loc=kloc, scale=scale)

    p_actual_ml_avg = p_actual_ml_total / ntrials
    p_actual_cp_avg = p_actual_cp_total / ntrials

    result = {
        'actual_p_ml' : np.ndarray.tolist(p_actual_ml_avg), 
        'actual_p_cp': np.ndarray.tolist(p_actual_cp_avg), 
        'p': np.ndarray.tolist(p)
        }

    return result


def reltest_gumbel(ntrials=100, nx=30, p=0.0001*np.asarray(range(1,10000)), loc=0, scale=1):
    '''
    Reliability test.

    Parameters
    ----------
    desired_p : array_like
        Probabilities at which to calculate quantiles.
    ntrials : int
        Number of trials to average over.
    nx : int
        Number of samples per trial.
    loc: float (default = 0)
        Loc parameter to test.
    scale: float (default = 1)
        Scale parameter to test.
    
    Returns
    -------
    dict
        Dictionary with keys:
            'actual_p_ml' : array_like
                Achieved probabilities using ML quantiles.
            'actual_p_cp' : array_like
                Achieved probabilities using CP quantiles.

    Each trial generates nx samples and calculates quantiles using the two methods.
    The difference between the methods is clearest when nx is in the range of 20-60.
    Increasing ntrials reduces the effect of random variations in the trials (100 is sufficient for many purposes).
    '''

    p_actual_ml_total = np.zeros(len(p))
    p_actual_cp_total = np.zeros(len(p))

    for i in range(ntrials):
        x = gumbel_r.rvs(loc=loc, scale=scale, size=nx)
        
        # calculate quantile using ML and CP methods
        # ML:
        ics = [loc, scale]
        loc_ml, scale_ml = minimize(lambda params: -cp_gumbel_libs.gumbel_logf(params, x), ics, method='BFGS').x 
        q_ml = gumbel_r.ppf(p, loc=loc_ml, scale=scale_ml)

        # CP:
        q_cp = cp_gumbel.ppf(x, p)['cp_quantiles']

        # feed back in for the actual probability
        p_actual_ml_total += gumbel_r.cdf(q_ml, loc=loc, scale=scale)
        p_actual_cp_total += gumbel_r.cdf(q_cp, loc=loc, scale=scale)

    p_actual_ml_avg = p_actual_ml_total / ntrials
    p_actual_cp_avg = p_actual_cp_total / ntrials

    result = {
        'actual_p_ml' : np.ndarray.tolist(p_actual_ml_avg), 
        'actual_p_cp': np.ndarray.tolist(p_actual_cp_avg), 
        'p': np.ndarray.tolist(p)
        }

    return result


# -----------------------------------------------------------------------------------------


def single_plot(data, ax):
    ax.plot(data['x'], data['y_ml'],  label='ML', color='red', linewidth=1)
    ax.plot(data['x'], data['y_cp'],  label='CP', color='blue')
    ax.plot(data['benchmark']['x'], data['benchmark']['y'], color='black', label='Benchmark')
    ax.set_xlim(data['limits']['x'])
    ax.set_ylim(data['limits']['y'])
    ax.set_xlabel(data['xlabel'])
    ax.set_ylabel(data['ylabel'])
    ax.set_title(data['title'])
    ax.legend()                         # crowds the graphs somewhat   



def plot(file):
    '''Passed the open file containing output from reltests, generates 5 graphs to compare ML and CP results.'''
    result = json.load(file)
    actual_p_ml = result['actual_p_ml']
    actual_p_cp = result['actual_p_cp']
    p = result['p']

    data = [
        reltest_libs.format_data_a(actual_p_ml, actual_p_cp, p), 
        reltest_libs.format_data_b(actual_p_ml, actual_p_cp, p), 
        reltest_libs.format_data_d(actual_p_ml, actual_p_cp, p), 
        reltest_libs.format_data_h(actual_p_ml, actual_p_cp, p), 
        reltest_libs.format_data_i(actual_p_ml, actual_p_cp, p)
        ]
    fig, axs = plt.subplots(3,2)
    single_plot(data[0], axs[0,0])
    single_plot(data[1], axs[0,1])
    single_plot(data[2], axs[1,0])
    single_plot(data[3], axs[1,1])
    single_plot(data[4], axs[2,0])
    fig.delaxes(axs[2,1])
    fig.tight_layout()
    plt.show()
