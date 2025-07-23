import numpy as np
import matplotlib.pyplot as plt
import pprint
from scipy.stats import genextreme
from scipy.optimize import minimize
import json
import warnings

import P110a_gev as cp_gev_a
import P110b_gev_libs as cp_gev_b

warnings.filterwarnings("ignore", category=RuntimeWarning)



def empirical(x):
    cdf = np.zeros(len(x))
    for i in range(len(x)):
        cdf[i] = np.sum(x <= x[i]) / (len(x)+1)
    plt.scatter(x, 1-cdf, label='empirical', color='black', marker='x')


def pdf_comparison(x, p):
    q = cp_gev_a.qgev_cp(x, p, pdf=True)
    plt.plot(q['cp_quantiles'], q['cp_pdf'], label='cp', color='red', linewidth=1)
    plt.plot(q['ml_quantiles'], q['ml_pdf'], label='ml', color='blue', linewidth=1)
    mu, sigma, xi = q['ml_params']
    plt.grid(True)
    plt.xlabel('q')
    plt.ylabel('pdf')
    plt.legend()
    plt.title(f'cp, ml gev comparison;  $\\xi={xi:.3f}$')   #x=[{x[0]},...,{x[-1]}];
    plt.show()
#pdf_comparison(x_2, p)


def cdf_comparison(x, p, edf=False):
    # plot cp, ml comparison
    q = cp_gev_a.qgev_cp(x, p)
    mu, sigma, xi = q['ml_params']
    if edf:
        plt.plot(q['cp_quantiles'], 1-p, label='cp', color='red', linewidth=1)
        plt.plot(q['ml_quantiles'], 1-p, label='ml', color='blue', linewidth=1)
    else:
        plt.plot(q['cp_quantiles'], p, label='cp', color='red', linewidth=1)
        plt.plot(q['ml_quantiles'], p, label='ml', color='blue', linewidth=1)
    plt.grid(True)
    plt.xlabel('q')
    plt.ylabel('p')
    plt.legend()
    plt.title(f'cp, ml gev comparison; $\\xi={xi:.3f}$')     #x=[{x[0]},...,{x[-1]}];
    plt.axhline(y=1, color='black', linewidth=1)
    plt.axhline(y=0, color='black', linewidth=1)
    plt.show()
#cdf_comparison(x, p, edf=True)



def reliability_test(desired_p, ntrials, nx):
    #returns dictionary with keys actual_p_ml, actual_p_cp (lists corresponding to entries in desired_p)
    xi = 0
    mu = 0
    sigma = 1

    p_actual_ml_total = np.zeros(len(desired_p))
    p_actual_cp_total = np.zeros(len(desired_p))

    for i in range(ntrials):
        x = genextreme.rvs(-xi, loc=mu, scale=sigma, size=nx)
        
        # calculate quantile using ML and CP methods
        # ML:
        ics = [mu, sigma, xi]
        mu_ml, sigma_ml, xi_ml = minimize(lambda params: -cp_gev_b.gev_loglik(params, x), ics, method='BFGS').x
        q_ml = genextreme.ppf(desired_p, -xi_ml, loc=mu_ml, scale=sigma_ml) 

        # CP:
        q_cp = cp_gev_a.qgev_cp(x, desired_p)['cp_quantiles']   

        # feed back in for the actual probability
        p_actual_ml_total += genextreme.cdf(q_ml, -xi, loc=mu, scale=sigma)
        p_actual_cp_total += genextreme.cdf(q_cp, -xi, loc=mu, scale=sigma)

    p_actual_ml_avg = p_actual_ml_total / ntrials
    p_actual_cp_avg = p_actual_cp_total / ntrials

    return {'actual_p_ml' : p_actual_ml_avg, 'actual_p_cp': p_actual_cp_avg}


ntrials = 100
nx = 40
p = 0.0001 *np.asarray( range(1, 10000))
result = reliability_test(p, ntrials, nx)
result['actual_p_ml'] = np.ndarray.tolist(result['actual_p_ml'])
result['actual_p_cp'] = np.ndarray.tolist(result['actual_p_cp'])
result['p'] = np.ndarray.tolist(p)
with open('test resltest nx=40, ntrials=1000, p=[0.0001, ..., 0.9999] step=0.0001.txt', 'w') as f:
    json.dump(result, f)


#pprint.pprint(q, width=160)