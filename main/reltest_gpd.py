import numpy as np
import pprint
from scipy.stats import genpareto
from scipy.optimize import minimize
import json
import warnings

import P120a_gpd_k1 as cp_gpd_a
import P120b_gpd_k1_libs as cp_gpd_b

warnings.filterwarnings("ignore", category=RuntimeWarning)


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
        q_cp = cp_gpd_a.qgpd_k1_cp(x, desired_p)['cp_quantiles']

        # feed back in for the actual probability
        p_actual_ml_total += genpareto.cdf(q_ml, -xi, loc=kloc, scale=sigma)
        p_actual_cp_total += genpareto.cdf(q_cp, -xi, loc=kloc, scale=sigma)

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
with open('reltest GPD 0', 'w') as f:
    json.dump(result, f)


#pprint.pprint(q, width=160)