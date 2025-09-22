import numpy as np
import scipy.stats
import scipy.optimize
from rusampling import Ru

import utils as cp_utils
import evaluate_dmgs_equation as cp_dmgs
import weibull_libs
import weibull_derivs
import reltest_libs



def ppf(x, p=np.arange(0.1, 1.0, 0.1), 
        means=False, waicscores=False, logscores=False, 
        dmgs=True, ru=False, ru_nsamples=100000,
        debug=False):
    """
    Weibull Distribution Predictions Based on a Calibrating Prior
    
    Parameters
    ----------
    x : array_like
        Data values
    p : array_like, optional
        Probabilities for quantiles (default 0.1 to 0.9 by 0.1)
    means : bool, optional
        Whether to calculate means (default False)
    waicscores : bool, optional
        Whether to calculate WAIC scores (default False)
    logscores : bool, optional
        Whether to calculate log scores (default False)
    dmgs : bool, optional
        Whether to use DMGS (asymptotic expansion for the integral) (default True)
    ru : bool, optional
        Whether to use Ratio of Uniforms sampling (default False)
    ru_nsamples : int, optional
        Number of Ratio of Uniforms samples, i.e. precision of the integral (default 100000)
    debug : bool, optional
        Whether to print debug messages (default False)
        
    Returns
    -------
    dict
        Dictionary containing results
    """
    # 1 intro
    debug = True
    debug = False
    
    # Input validation
    x = cp_utils.to_array(x)
    p = cp_utils.to_array(p)
    
    assert np.all(np.isfinite(x)) and np.all(~np.isnan(x)), "x must be finite and not NaN"
    assert np.all(np.isfinite(p)) and np.all(~np.isnan(p)), "p must be finite and not NaN"
    assert np.all(p > 0) and np.all(p < 1), "p must be between 0 and 1"
    assert np.all(x >= 0), "x must be non-negative"
    
    alpha = 1 - p
    nx = len(x)
    
    # 2 ml param estimate
    opt = scipy.optimize.minimize(lambda params: -weibull_libs.weibull_loglik(params, x), 
                                 [1, 1], method='BFGS')
    v1hat, v2hat = opt.x
    ml_params = np.array([v1hat, v2hat])
    
    if debug:
        print(f"  v1hat,v2hat={v1hat},{v2hat}")
    
    # 3 aic
    ml_value = -opt.fun
    maic = cp_utils.make_maic(ml_value, nparams=2)
    
    # 4 ml quantiles (vectorized over alpha)
    ml_quantiles = scipy.stats.weibull_min.ppf(1-alpha, c=v1hat, scale=v2hat)
    
    # 5 dmgs
    standard_errors = "ru not selected"
    rh_quantiles = "ru not selected"
    ru_quantiles = "ru not selected"
    waic1 = "ru not selected"
    waic2 = "ru not selected"
    ml_oos_logscore = "ru not selected"
    rh_oos_logscore = "ru not selected"
    cp_oos_logscore = "ru not selected"
    ml_mean = "ru not selected"
    rh_mean = "ru not selected"
    cp_mean = "ru not selected"
    cp_method = "ru not selected"
    
    if ru:
        # 6 lddi
        ldd = weibull_derivs.weibull_ldda(x, v1hat, v2hat)
        lddi = np.linalg.inv(ldd)
        standard_errors = cp_utils.make_se(nx, lddi)
        
        # 7 lddd
        if debug:
            print("  calculate lddd")
        lddd = weibull_derivs.weibull_lddda(x, v1hat, v2hat)
        
        # 7 mu1
        if debug:
            print("calculate mu1")
        mu1 = weibull_derivs.weibull_mu1fa(alpha, v1hat, v2hat)
        
        # 8 mu2
        if debug:
            print("calculate mu2")
        mu2 = weibull_derivs.weibull_mu2fa(alpha, v1hat, v2hat)
        
        # 9 q rhp
        if debug:
            print("  rhp")
        lambdad_rhp = np.array([-1/v1hat, -1/v2hat])
        
        # 10 derive the bayesian dq based on v2hat
        if debug:
            print("  fhat, dq and rhp quantiles")
        fhat = scipy.stats.weibull_min.pdf(ml_quantiles, c=v1hat, scale=v2hat)
        dq = cp_dmgs.dmgs(lddi, lddd, mu1, lambdad_rhp, mu2, dim=2)
        rh_quantiles = ml_quantiles + dq / (nx * fhat)
        
        # 11 means
        means_result = weibull_libs.weibull_means(means, ml_params, lddi, lddd, 
                                                  lambdad_rhp, nx, dim=2)
        ml_mean = means_result['ml_mean']
        rh_mean = means_result['rh_mean']
        
        # 12 waicscores
        waic = weibull_libs.weibull_waic(waicscores, x, v1hat, v2hat, lddi, lddd, 
                                        lambdad_rhp)
        waic1 = waic['waic1']
        waic2 = waic['waic2']
        
        # 13 logscores
        logscores_result = weibull_libs.weibull_logscores(logscores, x)
        ml_oos_logscore = logscores_result['ml_oos_logscore']
        rh_oos_logscore = logscores_result['rh_oos_logscore']
        
        # 14 ru
        ru_quantiles = "integrate_sampling not selected"
        if ru:
            rusim = rvs(ru_nsamples, x, ru=True, mlcp=False)
            ru_quantiles = cp_utils.makeq(rusim['ru_deviates'], p)
    
    # return
    return {
        'ml_params': ml_params,
        'ml_value': ml_value,
        'standard_errors': standard_errors,
        'ml_quantiles': ml_quantiles,
        'cp_quantiles': rh_quantiles,
        'ru_quantiles': ru_quantiles,
        'maic': maic,
        'waic1': waic1,
        'waic2': waic2,
        'ml_oos_logscore': ml_oos_logscore,
        'cp_oos_logscore': rh_oos_logscore,
        'ml_mean': ml_mean,
        'cp_mean': rh_mean,
        'cp_method': cp_utils.rhp_dmgs_cpmethod()
    }

def rvs(n, x, ru=False, mlcp=True, debug=False):
    """
    Random generation for Weibull CP
    
    Parameters
    ----------
    n : int
        Number of samples
    x : array_like
        Data for parameter estimation
    ru : bool, optional
        Whether to use Ratio of Uniforms sampling (default False)
    mlcp : bool, optional
        Whether to use ML/CP approach (default True)
    debug : bool, optional
        Whether to print debug messages (default False)
        
    Returns
    -------
    dict
        Dictionary containing generated samples
    """
    # Input validation
    x = cp_utils.to_array(x)

    assert np.all(np.isfinite(x)) and np.all(~np.isnan(x)), "x must be finite and not NaN"
    assert np.all(x >= 0), "x must be non-negative"
    
    ml_params = "mlcp not selected"
    ml_deviates = "mlcp not selected"
    cp_deviates = "mlcp not selected"
    ru_deviates = "ru not selected"
    
    if mlcp:
        q = ppf(x, np.random.uniform(0, 1, n))
        ml_params = q['ml_params']
        ml_deviates = q['ml_quantiles']
        cp_deviates = q['cp_quantiles']
    
    if ru:
        th = tsf(n, x)['theta_samples']
        ru_deviates = np.zeros(n)
        for i in range(n):
            ru_deviates[i] = scipy.stats.weibull_min.rvs(c=th[i, 0], scale=th[i, 1], size=1)
    
    op = {
        'ml_params': ml_params,
        'ml_deviates': ml_deviates,
        'cp_deviates': cp_deviates,
        'ru_deviates': ru_deviates,
        'cp_method': cp_utils.rhp_dmgs_cpmethod()
    }
    return op

def pdf(x, y=None, ru=False, ru_nsamples=1000, debug=False):
    """
    Density function for Weibull CP
    
    Parameters
    ----------
    x : array_like
        Data for parameter estimation
    y : array_like, optional
        Points to evaluate density (default same as x)
    ru : bool, optional
        Whether to use Ratio of Uniforms sampling (default False)
    ru_nsamples : int, optional
        Number of Ratio of Uniforms samples (default 1000)
    debug : bool, optional
        Whether to print debug messages (default False)
        
    Returns
    -------
    dict
        Dictionary containing density values
    """

    x = cp_utils.to_array(x)
    if y is None:
        y = x
    else:
        y = cp_utils.to_array(y)
    
    assert np.all(np.isfinite(x)) and np.all(~np.isnan(x)), "x must be finite and not NaN"
    assert np.all(np.isfinite(y)) and np.all(~np.isnan(y)), "y must be finite and not NaN"
    assert np.all(x >= 0), "x must be non-negative"
    assert np.all(y >= 0), "y must be non-negative"
    
    dd = weibull_libs.dweibullsub(x=x, y=y)
    ru_pdf = "ru not selected"
    
    if ru:
        th = tsf(ru_nsamples, x)['theta_samples']
        ru_pdf = np.zeros(len(y))
        for ir in range(ru_nsamples):
            ru_pdf += scipy.stats.weibull_min.pdf(y, c=th[ir, 0], scale=th[ir, 1])
        ru_pdf = ru_pdf / ru_nsamples
    
    op = {
        'ml_params': dd['ml_params'],
        'ml_pdf': dd['ml_pdf'],
        'cp_pdf': dd['rh_pdf'],
        'ru_pdf': ru_pdf,
        'cp_method': cp_utils.rhp_dmgs_cpmethod()
    }
    return op

def cdf(x, y=None, ru=False, ru_nsamples=1000, debug=False):
    """
    Distribution function for Weibull CP
    
    Parameters
    ----------
    x : array_like
        Data for parameter estimation
    y : array_like, optional
        Points to evaluate CDF (default same as x)
    ru : bool, optional
        Whether to use Ratio of Uniforms sampling (default False)
    ru_nsamples : int, optional
        Number of Ratio of Uniforms samples (default 1000)
    debug : bool, optional
        Whether to print debug messages (default False)
        
    Returns
    -------
    dict
        Dictionary containing CDF values
    """
    x = cp_utils.to_array(x)
    if y is None:
        y = x
    else:
        y = cp_utils.to_array(y)

    assert np.all(np.isfinite(x)) and np.all(~np.isnan(x)), "x must be finite and not NaN"
    assert np.all(np.isfinite(y)) and np.all(~np.isnan(y)), "y must be finite and not NaN"
    assert np.all(x >= 0), "x must be non-negative"
    assert np.all(y >= 0), "y must be non-negative"
    
    dd = weibull_libs.dweibullsub(x=x, y=y)
    ru_cdf = "ru not selected"
    
    if ru:
        th = tsf(ru_nsamples, x)['theta_samples']
        ru_cdf = np.zeros(len(y))
        for ir in range(ru_nsamples):
            ru_cdf += scipy.stats.weibull_min.cdf(y, c=th[ir, 0], scale=th[ir, 1])
        ru_cdf = ru_cdf / ru_nsamples
    
    op = {
        'ml_params': dd['ml_params'],
        'ml_cdf': dd['ml_cdf'],
        'cp_cdf': dd['rh_cdf'],
        'ru_cdf': ru_cdf,
        'cp_method': cp_utils.rhp_dmgs_cpmethod()
    }
    return op

def tsf(n, x, debug=False):
    """
    Theta sampling for Weibull CP
    
    Parameters
    ----------
    n : int
        Number of samples
    x : array_like
        Data values
        
    Returns
    -------
    array of float
        Theta samples.
    """
    x = cp_utils.to_array(x)

    assert np.all(np.isfinite(x)) and np.all(~np.isnan(x)), "x must be finite and not NaN"
    assert np.all(x >= 0), "x must be non-negative"

    ml_params = scipy.optimize.minimize(
        lambda params: -weibull_libs.weibull_loglik(params, x), [1, 1], method='BFGS').x
    
    t = Ru(weibull_libs.weibull_logf, x=x, d=2, ics=ml_params)
    if debug:
        t.info()
    
    return {'theta_samples': t.rvs(n=n)}


def reltest(plot=True, ntrials=50, nx=30, p=0.0001*np.asarray(range(1,10000)), shape=1, scale=1, plot_option='tail'):
    '''
    Reliability test.

    Parameters
    ----------
    plot: bool (default = True)
        Create a plot of the results immediately.
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
    plot_option: str (default='tail')
        For use when plot=True, determines which graph to output.
        Options are 'unformatted', 'b', 'd', 'tail', 'i', 'all'.
        'tail' demonstrates tail probabilities the best.
    
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
        x = scipy.stats.weibull_min.rvs(c=shape, scale=scale, size=nx)

        info_cp = ppf(x, p)
        q_cp = info_cp['cp_quantiles']
        q_ml = info_cp['ml_quantiles']

        # feed back in for the actual probability
        p_actual_ml_total += scipy.stats.weibull_min.cdf(q_ml, c=shape, scale=scale)
        p_actual_cp_total += scipy.stats.weibull_min.cdf(q_cp, c=shape, scale=scale)

    p_actual_ml_avg = p_actual_ml_total / ntrials
    p_actual_cp_avg = p_actual_cp_total / ntrials

    result = {
        'actual_p_ml' : np.ndarray.tolist(p_actual_ml_avg), 
        'actual_p_cp': np.ndarray.tolist(p_actual_cp_avg), 
        'p': np.ndarray.tolist(p)
        }
    
    if plot:
        reltest_libs.plot(result, plot_option)

    return result