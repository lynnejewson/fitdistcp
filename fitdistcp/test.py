import numpy as np
from pprint import pprint

import fitdistcp.expon

import norm as cp_norm
import lnorm as cp_lnorm
import gumbel as cp_gumbel
import gamma as cp_gamma
import weibull as cp_weibull
import expon as cp_expon
import genextreme as cp_genextreme
import genextreme_p1 as cp_genextreme_p1
import genextreme_p12 as cp_genextreme_p12
import genpareto as cp_genpareto
import test_example_data as data


def ppf():
    q = [
        cp_norm.ppf(data.norm[1], logscores=True, unbiasedv=True, waicscores=True),
        cp_lnorm.ppf(data.lnorm[1], logscores=True, means= True, waicscores=True),
        cp_gamma.ppf(data.gamma[1], means=True, waicscores=True),
        cp_gumbel.ppf(data.gumbel[1], logscores=True, means=True, waicscores=True),
        cp_weibull.ppf(data.weibull[2], logscores=True, means=True, waicscores=True),
        cp_expon.ppf(data.expon[1], logscores=True, means=True, waicscores=True),
        cp_genextreme.ppf(data.gev[0], means=True, waicscores=True),
        cp_genextreme_p1.ppf(data.gev_p1[0], data.gev_p1_t[0], t0=21, means=True, waicscores=True),
        cp_genpareto.ppf(data.gpd[0], means=True, waicscores=True),
        cp_genextreme_p12.ppf(data.gev_p12[0], data.gev_p12_t1[0], data.gev_p12_t2[0], t01=0, t02=0)
    ]
    return q

def rvs():
    y = [
        cp_norm.rvs(1, data.norm[0]),
        cp_lnorm.rvs(1, data.lnorm[0]),
        cp_gamma.rvs(1, data.gamma[0]),
        cp_gumbel.rvs(1, data.gumbel[0]),
        cp_weibull.rvs(1, data.weibull[0]),
        cp_expon.rvs(1, data.expon[0]),
        cp_genextreme.rvs(1, data.gev[0]),
        cp_genextreme_p1.rvs(1, data.gev_p1[0], data.gev_p1_t[0], t0=21),
        cp_genpareto.rvs(1, data.gpd[0]),
        cp_genextreme_p12.rvs(1, data.gev_p12[0], data.gev_p12_t1[0], data.gev_p12_t2[0], t01=0, t02=0)
    ]
    return y

def pdf():
    d = [
        cp_norm.pdf(data.norm[0], y=[0]),
        cp_lnorm.pdf(data.lnorm[0], y=[0]),
        cp_gamma.pdf(data.gamma[0], y=[0]),
        cp_gumbel.pdf(data.gumbel[0], y=[1]),
        cp_weibull.pdf(data.weibull[0], y=[1]),
        cp_expon.pdf(data.expon[0], y=[1]),
        cp_genextreme.pdf(data.gev[0], y=[1]),
        cp_genextreme_p1.pdf(data.gev_p1[0], data.gev_p1_t[0], y=[1], t0=21),
        cp_genpareto.pdf(data.gpd[0], y=[1]),
        cp_genextreme_p12.pdf(data.gev_p12[0], data.gev_p12_t1[0], data.gev_p12_t2[0], t01=0, t02=0)
    ]
    return d

def cdf():
    q = [
        cp_norm.cdf(data.norm[0]),
        cp_lnorm.cdf(data.lnorm[0]),
        cp_gamma.cdf(data.gamma[0]),
        cp_gumbel.cdf(data.gumbel[0]),
        cp_weibull.cdf(data.weibull[0]),
        cp_expon.cdf(data.expon[0]),
        cp_genextreme.cdf(data.gev[0]),
        cp_genextreme_p1.cdf(data.gev_p1[0], data.gev_p1_t[0], t0=21),
        cp_genpareto.cdf(data.gpd[0]),
        cp_genextreme_p12.cdf(data.gev_p12[0], data.gev_p12_t1[0], data.gev_p12_t2[0], t01=0, t02=0)
    ]
    return q


# TSF
n = 10
r = [
    #cp_norm.tsf(n, data.norm[0]),
    #cp_lnorm.tsf(n, data.lnorm[0]),
    #cp_gamma.tsf(n, data.gamma[1]),
    #cp_gumbel.tsf(n, data.gumbel[0]),
    #cp_weibull.tsf(n, data.weibull[0]),
    #cp_expon.tsf(n, data.expon[0]),
    #cp_genextreme.tsf(n, data.gev[0]),
    #cp_genpareto.tsf(n, data.gpd[0]),
    #cp_genextreme_p1.tsf(n, data.gev_p1[0], data.gev_p1_t[0]),
    #cp_genextreme_p12.tsf(n, data.gev_p12[0], data.gev_p12_t1[0], data.gev_p12_t2[0])
]