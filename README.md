fitdistcp

fitdistcp is a free Python package for fitting statistical models using calibrating priors, with the goal of making reliable predictions. It is an alternative to packages such as scipy.stats.genextreme, which use the maximum likelihood method, and underestimate predictive tail probabilities.

Fitdistcp is based on the results from Reducing Reliability Bias in Assessments of Extreme Weather Risk using Calibrating Priors, S. Jewson, T. Sweeting and L. Jewson (2024): https://doi.org/10.5194/ascmo-11-1-2025.
 
More information and examples are available at https://www.fitdistcp.info/index.html, including the equivalent (extended) package in R.
Install fitdistcp from Pypi using pip install fitdistcp.

 

There are four functions associated with each distribution. Each function accepts the sample data x as a parameter and returns a dict of the relevant results.

-        q: returns quantiles calculated using the CP method.

-        R: returns random samples from the distribution, estimated using the CP method

-        …

Tests

-        The cdf and pdf can be estimated for a set of data, using both the ML and CP method, and plotted.

-        Reliability tests are provided. A reliability test involves generating a sample from a known distribution, with known quantiles, and separately calculates these quantiles using the ML and CP methods. The actual quantiles can be compared to the estimated quantiles in a number of different ways which are demonstrated in the package.

 

Example: Fitting a GEV distribution

> import fitdistcp

> import numpy as np

> import scipy.stats

> import matplotlib.pyplot as plt

> x=genextreme.rvs(0,size=20)                                               # make some example training data 
> p=np.range(0.001,0.999,0.001)                              # define the probabilities at which we wish to calculate the quantiles
> q=qgev_cp(x,p)                                             # this command calculates two sets of predictive quantiles for the GEV, 
                                                                            # one based on maxlik, and one that includes parameter uncertainty based on a calibrating prior
> print(q[‘ml_params’])                                                     # have a look at the maxlik parameters
> plt.plot(q[‘ml_quantiles’],p)                              # plot the maxlik quantiles
> plt.plot(q[‘cp_quantiles’],p,’o’,color="red")              # overplot the quantiles that include parameter uncertainty, which will have fatter tails
> plt.show()