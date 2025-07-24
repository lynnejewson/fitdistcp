import numpy as np
import pprint
from scipy.stats import genpareto
from scipy.optimize import minimize
import json
import warnings
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.getcwd() + '\\lib')

import genpareto as cp_gpd_a
import genpareto_libs as cp_gpd_b
import tests_lib


warnings.filterwarnings("ignore", category=RuntimeWarning)



def pdf_comparison():
    x = genpareto.rvs(0.3, loc=0, scale=1, size=20)
    p = 0.0005*np.asarray(range(1,2000))
    q = cp_gpd_a.ppf(x, p, pdf=True)
    print(q)
    sigma, xi = q['ml_params']
    actual_p = genpareto.pdf(q['cp_quantiles'], 0.3, loc=0, scale=1)
    plt.plot(q['cp_quantiles'], actual_p, label='sampling distribution', color='black', linewidth=1)
    tests_lib.pdf_comparison(q, title=f'cp, ml GPD comparison;  $\\xi={xi:.3f}$')


def cdf_comparison(edf=False):
    x = genpareto.rvs(0.3, loc=5, scale=1, size=50)
    p = 0.0005*np.asarray(range(1,2000))
    q = cp_gpd_a.ppf(x, p)
    sigma, xi = q['ml_params']
    tests_lib.cdf_comparison(q, p, title=f'cp, ml GPD comparison;  $\\xi={xi:.3f}$')


def reliability_test(desired_p, ntrials, nx):
    #returns dictionary with keys actual_p_ml, actual_p_cp (lists corresponding to entries in desired_p)
    kloc = 0
    sigma = 1
    xi = 0

    p_actual_ml_total = np.zeros(len(desired_p))
    p_actual_cp_total = np.zeros(len(desired_p))

    for i in range(ntrials):
        x = genpareto.rvs(-xi, loc=kloc, scale=sigma, size=nx)
        
        # calculate quantile using ML and CP methods
        # ML:
        ics = [sigma, xi]
        sigma_ml, xi_ml = minimize(lambda params: -cp_gpd_b.gpd_k1_loglik(params, x, kloc), ics, method='BFGS').x
        q_ml = genpareto.ppf(desired_p, -xi_ml, loc=kloc, scale=sigma_ml)

        # CP:
        q_cp = cp_gpd_a.ppf(x, desired_p)['cp_quantiles']

        # feed back in for the actual probability
        p_actual_ml_total += genpareto.cdf(q_ml, -xi, loc=kloc, scale=sigma)
        p_actual_cp_total += genpareto.cdf(q_cp, -xi, loc=kloc, scale=sigma)

    p_actual_ml_avg = p_actual_ml_total / ntrials
    p_actual_cp_avg = p_actual_cp_total / ntrials

    return {'actual_p_ml' : p_actual_ml_avg, 'actual_p_cp': p_actual_cp_avg}


def reltest_to_file():
    ntrials = 100
    nx = 40
    p = 0.0001 *np.asarray( range(1, 10000))
    result = reliability_test(p, ntrials, nx)
    result['actual_p_ml'] = np.ndarray.tolist(result['actual_p_ml'])
    result['actual_p_cp'] = np.ndarray.tolist(result['actual_p_cp'])
    result['p'] = np.ndarray.tolist(p)
    file_name = "reltest GPD 4.txt"
    file_path = os.getcwd() + '\\reltest output\\' + file_name
    with open(file_path, 'w') as f:
        json.dump(result, f)


x_example_1 = [
    0.91794411, 0.29501841, 5.22681723, 0.44249577, 0.38225736, 1.65562589,
    0.33388235, 0.29924417, 0.20876138, 4.18325104, 2.22945457, 1.45345549,
    0.31319118, 0.01162536, 0.13348783, 0.38851889, 0.32642607, 0.72000737,
    1.06938394, 2.52318835
]
x_example_2 = [
    0.20572726, 1.86384593, 0.48910372, 1.43486757, 1.159838, 0.15611357,
    0.94287216, 0.03497607, 0.5713605,  1.1172626,  0.6266541,  1.23199102, 0.96530044,
    2.0140288,  0.37606083, 0.62832804, 1.07610164, 1.15995558, 0.10547796, 0.65184389
]
x = genpareto.rvs(0, loc=0, scale=1, size=20)
q = cp_gpd_a.qgpd_k1_cp(x_example_1)
pprint.pprint(q)

#reltest_to_file()
#pdf_comparison()
#cdf_comparison()