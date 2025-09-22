import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize
from rusampling import Ru

import utils as cp_utils
import evaluate_dmgs_equation as cp_dmgs
import gamma_libs
import gamma_derivs
import reltest_libs


def ppf(x, p=None, fd1=0.01, fd2=0.01, means=False, waicscores=False, 
              logscores=False, 
              dmgs=True, ru=False, ru_nsamples=100000,
              prior="type 1", 
              debug=False, aderivs=True):
    """
    Gamma Distribution Quantile Predictions Based on a Calibrating Prior
    
    Parameters
    ----------
    x : array-like
        Training data values
    p : array-like, optional
        Probabilities for quantiles (default: seq(0.1,0.9,0.1))
    fd1 : float
        Step size for v1 finite differences (default 0.01)
    fd2 : float  
        Step size for v2 finite differences (default 0.01)
    means : bool
        Whether to calculate means (default False)
    waicscores : bool
        Whether to calculate WAIC scores (default False)
    logscores : bool
        Whether to calculate log scores (default False)
    dmgs : bool
        Whether to use DMGS integration (default True)
    ru : bool
        Whether to use Ratio of Uniforms sampling method to integrate (default False)
    ru_nsamples : int
        Number of Ratio of Uniforms samples (default 100000)
    prior : str
        Prior type ("type 1" or "type 2", default "type 1")
    debug : bool
        Debug flag (default False)
    aderivs : bool
        Whether to use analytical derivatives (default True)
        
    Returns
    -------
    dict
        Dictionary with ml_params, ml_value, standard_errors, ml_quantiles, 
        cp_quantiles, ru_quantiles, maic, waic1, waic2, ml_oos_logscore,
        cp_oos_logscore, ml_mean, cp_mean, cp_method
    """
    
    # 1 intro
    debug = True
    debug = False
    
    if p is None:
        p = np.arange(0.1, 1.0, 0.1)
    
    x = np.array(x)
    p = np.array(p)
    
    assert np.all(np.isfinite(x)), "x must be finite"
    assert np.all(~np.isnan(x)), "x must not contain NaN"
    assert np.all(np.isfinite(p)), "p must be finite" 
    assert np.all(~np.isnan(p)), "p must not contain NaN"
    assert np.all(p > 0), "p must be > 0"
    assert np.all(p < 1), "p must be < 1"
    assert np.all(x >= 0), "x must be >= 0"
    
    alpha = 1 - p
    nx = len(x)
    nalpha = len(alpha)
    
    # 2 ml param estimate
    opt = optimize.minimize(lambda params: -gamma_libs.gamma_loglik(params, x), 
                          x0=[1, 1], method='BFGS')
    v1hat = opt.x[0]
    v2hat = opt.x[1]
    ml_params = np.array([v1hat, v2hat])
    
    if debug:
        print(f"  v1hat,v2hat={v1hat},{v2hat}")
    
    # 3 aic
    ml_value = -opt.fun
    maic = cp_utils.make_maic(ml_value, nparams=2)
    
    # 4 ml quantiles (vectorized over alpha)
    ml_quantiles = stats.gamma.ppf(1-alpha, a=v1hat, scale=v2hat)
    
    # test of gg code (for future implementation of mpd theory, as a test of the mpd code)
    # gamma_gg(v1hat,fd1,v2hat,fd2)
    
    # 5 dmgs
    standard_errors = "dmgs not selected"
    cp_quantiles = "dmgs not selected"
    ru_quantiles = "dmgs not selected"
    waic1 = "dmgs not selected"
    waic2 = "dmgs not selected"
    ml_oos_logscore = "dmgs not selected"
    cp_oos_logscore = "dmgs not selected"
    cp_oos_logscore = "dmgs not selected"
    ml_mean = "dmgs not selected"
    cp_mean = "dmgs not selected"
    cp_mean = "dmgs not selected"
    cp_method = "dmgs not selected"
    
    if dmgs:
        # 6 lddi
        if aderivs:
            ldd = gamma_derivs.gamma_ldda(x, v1hat, v2hat)
        if not aderivs:
            ldd = gamma_derivs.gamma_ldd(x, v1hat, fd1, v2hat, fd2)
        lddi = np.linalg.inv(ldd)
        standard_errors = cp_utils.make_se(nx, lddi)
        
        # 7 lddd
        if debug:
            print("  calculate lddd")
        if aderivs:
            lddd = gamma_derivs.gamma_lddda(x, v1hat, v2hat)
        if not aderivs:
            lddd = gamma_derivs.gamma_lddd(x, v1hat, fd1, v2hat, fd2)
        
        # 8 mu1
        if debug:
            print("calculate mu1")
        mu1 = gamma_libs.gamma_mu1f(alpha, v1hat, fd1, v2hat, fd2)
        
        # 9 mu2
        if debug:
            print("calculate mu2")
        mu2 = gamma_libs.gamma_mu2f(alpha, v1hat, fd1, v2hat, fd2)
        
        # 10 q cp
        # (I compared these two priors, and the double prior worked better)
        # (so I made the better one type 1)
        # (see the actuary paper)
        if debug:
            print("  cp")
        if prior == "type 1":
            lambdad_cp = np.array([-1/v1hat, -1/v2hat])
        elif prior == "type 2":
            lambdad_cp = np.array([0, -1/v2hat])
        else:
            print("invalid prior choice.")
            raise ValueError("invalid prior choice.")
        
        # lambdad_cp=c(0,-1/v2hat)               #this worked ok
        # lambdad_cp=c(-1/v1hat,-1/v2hat)        #but this worked better
        
        # 11 derive the bayesian dq based on v2hat
        if debug:
            print("  fhat, dq and cp quantiles")
        fhat = stats.gamma.pdf(ml_quantiles, a=v1hat, scale=v2hat)
        dq = cp_dmgs.dmgs(lddi, lddd, mu1, lambdad_cp, mu2, dim=2)
        cp_quantiles = ml_quantiles + dq/(nx*fhat)
        
        # 12 means
        means_result = gamma_libs.gamma_means(means, ml_params, lddi, lddd, lambdad_cp, nx, dim=2)
        ml_mean = means_result['ml_mean']
        cp_mean = means_result['cp_mean']
        
        # 13 waicscores
        waic = gamma_libs.gamma_waic(waicscores, x, v1hat, fd1, v2hat, fd2, lddi, lddd, 
                         lambdad_cp, aderivs)
        waic1 = waic['waic1']
        waic2 = waic['waic2']
        
        # 14 logscores
        logscores_result = gamma_libs.gamma_logscores(logscores, x, fd1, fd2, aderivs)
        ml_oos_logscore = logscores_result['ml_oos_logscore']
        cp_oos_logscore = logscores_result['cp_oos_logscore']
        
        # 15 ru
        ru_quantiles = "ru not selected"
        if ru:
            rusim = rvs(ru_nsamples, x, ru=True, mlcp=False)
            ru_quantiles = cp_utils.makeq(rusim['ru_deviates'], p)
            
    # end of if dmgs
    
    # return
    return {
        'ml_params': ml_params,
        'ml_value': ml_value,
        # 'ldd': ldd,
        # 'lddi': lddi,
        # 'expinfmat': expinfmat,
        # 'expinfmati': expinfmati,
        'standard_errors': standard_errors,
        'ml_quantiles': ml_quantiles,
        'cp_quantiles': cp_quantiles,
        'ru_quantiles': ru_quantiles,
        'maic': maic,
        'waic1': waic1,
        'waic2': waic2,
        'ml_oos_logscore': ml_oos_logscore,
        'cp_oos_logscore': cp_oos_logscore,
        'ml_mean': ml_mean,
        'cp_mean': cp_mean,
        'cp_method': cp_utils.adhoc_dmgs_cpmethod()
    }


def rvs(n, x, fd1=0.01, fd2=0.01, ru=False, mlcp=True, debug=False, aderivs=True):
    """
    Random number generation for Gamma Distribution with Calibrating Prior
    
    Parameters
    ----------
    n : int
        Number of samples to generate
    x : array-like
        Training data values
    fd1 : float
        Step size for v1 finite differences (default 0.01)
    fd2 : float
        Step size for v2 finite differences (default 0.01)
    ru : bool
        Whether to use Ratio of Uniforms sampling (default False)
    mlcp : bool
        Whether to use ML/CP sampling (default True)
    debug : bool
        Debug flag (default False)
    aderivs : bool
        Whether to use analytical derivatives (default True)
        
    Returns
    -------
    dict
        Dictionary with ml_params, ml_deviates, cp_deviates, ru_deviates, cp_method
    """
    
    # stopifnot(is.finite(n),!is.na(n),is.finite(x),!is.na(x),!x<0)
    x = np.array(x)
    assert np.all(np.isfinite(x)), "x must be finite"
    assert np.all(~np.isnan(x)), "x must not contain NaN"
    assert np.all(x >= 0), "x must be >= 0"
    
    ml_params = "mlcp not selected"
    ml_deviates = "mlcp not selected"
    cp_deviates = "mlcp not selected"
    ru_deviates = "ru not selected"
    
    if mlcp:
        q = ppf(x, np.random.uniform(0, 1, n), fd1=fd1, fd2=fd2, aderivs=aderivs)
        ml_params = q['ml_params']
        ml_deviates = q['ml_quantiles']
        cp_deviates = q['cp_quantiles']
    
    if ru:
        th = tsf(n, x)['theta_samples']
        ru_deviates = np.zeros(n)
        for i in range(n):
            ru_deviates[i] = np.random.gamma(shape=th[i,0], scale=th[i,1])
    
    op = {
        'ml_params': ml_params,
        'ml_deviates': ml_deviates,
        'cp_deviates': cp_deviates,
        'ru_deviates': ru_deviates,
        'cp_method': cp_utils.adhoc_dmgs_cpmethod()
    }
    return op


def pdf(x, y=None, fd1=0.01, fd2=0.01, ru=False, ru_nsamples=1000, debug=False, aderivs=True):
    """
    Density function for Gamma Distribution with Calibrating Prior
    
    Parameters
    ----------
    x : array-like
        Training data values
    y : array-like, optional
        Evaluation points (default: same as x)
    fd1 : float
        Step size for v1 finite differences (default 0.01)
    fd2 : float
        Step size for v2 finite differences (default 0.01)
    ru : bool
        Whether to use Ratio of Uniforms sampling (default False)
    ru_nsamples : int
        Number of Ratio of Uniforms samples (default 1000)
    debug : bool
        Debug flag (default False)
    aderivs : bool
        Whether to use analytical derivatives (default True)
        
    Returns
    -------
    dict
        Dictionary with ml_params, ml_pdf, cp_pdf, ru_pdf, cp_method
    """

    x = cp_utils.to_array(x)
    if y is None:
        y = x
    else:
        y = cp_utils.to_array(y)
    
    assert np.all(np.isfinite(x)), "x must be finite"
    assert np.all(~np.isnan(x)), "x must not contain NaN"
    assert np.all(np.isfinite(y)), "y must be finite"
    assert np.all(~np.isnan(y)), "y must not contain NaN"
    assert np.all(x >= 0), "x must be >= 0"
    assert np.all(y >= 0), "y must be >= 0"
    
    dd = gamma_libs.dgammasub(x=x, y=y, fd1=fd1, fd2=fd2, aderivs=aderivs)
    ru_pdf = "ru not selected"
    
    if ru:
        th = tsf(ru_nsamples, x)['theta_samples']
        ru_pdf = np.zeros(len(y))
        for ir in range(ru_nsamples):
            ru_pdf = ru_pdf + stats.gamma.pdf(y, a=th[ir,0], scale=th[ir,1])
        ru_pdf = ru_pdf / ru_nsamples
    
    op = {
        'ml_params': dd['ml_params'],
        'ml_pdf': dd['ml_pdf'],
        'cp_pdf': dd['cp_pdf'],
        'ru_pdf': ru_pdf,
        'cp_method': cp_utils.adhoc_dmgs_cpmethod()
    }
    return op

def cdf(x, y=None, fd1=0.01, fd2=0.01, ru=False, ru_nsamples=1000, debug=False, aderivs=True):
    """
    CDF function for Gamma Distribution with Calibrating Prior
    
    Parameters
    ----------
    x : array-like
        Training data values
    y : array-like, optional
        Evaluation points (default: same as x)
    fd1 : float
        Step size for v1 finite differences (default 0.01)
    fd2 : float
        Step size for v2 finite differences (default 0.01)
    ru : bool
        Whether to use Ratio of Uniforms sampling (default False)
    ru_nsamples : int
        Number of Ratio of Uniforms samples (default 1000)
    debug : bool
        Debug flag (default False)
    aderivs : bool
        Whether to use analytical derivatives (default True)
        
    Returns
    -------
    dict
        Dictionary with ml_params, ml_cdf, cp_cdf, ru_cdf, cp_method
    """
    
    if y is None:
        y = x
    
    x = np.array(x)
    y = np.array(y)
    
    assert np.all(np.isfinite(x)), "x must be finite"
    assert np.all(~np.isnan(x)), "x must not contain NaN"
    assert np.all(np.isfinite(y)), "y must be finite"
    assert np.all(~np.isnan(y)), "y must not contain NaN"
    assert np.all(x >= 0), "x must be >= 0"
    assert np.all(y >= 0), "y must be >= 0"
    
    dd = gamma_libs.dgammasub(x=x, y=y, fd1=fd1, fd2=fd2, aderivs=aderivs)
    ru_cdf = "ru not selected"
    
    if ru:
        th = tsf(ru_nsamples, x)['theta_samples']
        ru_cdf = np.zeros(len(y))
        for ir in range(ru_nsamples):
            ru_cdf = ru_cdf + stats.gamma.cdf(y, a=th[ir,0], scale=th[ir,1])
        ru_cdf = ru_cdf / ru_nsamples
    
    op = {
        'ml_params': dd['ml_params'],
        'ml_cdf': dd['ml_cdf'],
        'cp_cdf': dd['cp_cdf'],
        'ru_cdf': ru_cdf,
        'cp_method': cp_utils.adhoc_dmgs_cpmethod()
    }
    return op


def tsf(n, x):
    """
    Theta sampling for Gamma Distribution with Calibrating Prior
    
    Parameters
    ----------
    n : int
        Number of theta samples to generate
    x : array-like
        Training data values
        
    Returns
    -------
    array of float
        Array of theta_samples
    """
    
    x = cp_utils.to_array(x)
    assert np.all(np.isfinite(x)), "x must be finite"
    assert np.all(~np.isnan(x)), "x must not contain NaN"
    assert np.all(x >= 0), "x must be >= 0"

    ml_params = optimize.minimize(lambda params: -gamma_libs.gamma_loglik(params, x), 
                          x0=[1, 1], method='BFGS').x

    # ics shape, scale
    t = Ru(gamma_libs.gamma_logf, x=x, d=2, ics=ml_params)
    if t.mode[0] < 1:
        print('Shape parameter mode < 1. May cause numerical issues.')
    
    return {'theta_samples': t.rvs(n=n)}


def reltest(plot=True, ntrials=50, nx=30, p=0.0001*np.asarray(range(1,10000)), a=2, scale=1, plot_option='tail'):
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
    a: float (default = 1)
        Scale parameter to test.
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
        x = stats.gamma.rvs(a, scale=scale, size=nx)

        info_cp = ppf(x, p)
        q_cp = info_cp['cp_quantiles']
        q_ml = info_cp['ml_quantiles']

        # feed back in for the actual probability
        p_actual_ml_total += stats.gamma.cdf(q_ml, a, scale=scale)
        p_actual_cp_total += stats.gamma.cdf(q_cp, a, scale=scale)

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