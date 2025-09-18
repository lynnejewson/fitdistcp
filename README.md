# fitdistcp

fitdistcp is a free Python package for fitting statistical models using calibrating priors, with the goal of making reliable predictions. Install using >pip install fitdistcp.

fitdistcp implements the method developed in *Reducing Reliability Bias in Assessments of Extreme Weather Risk using Calibrating Priors*, S. Jewson, T. Sweeting and L. Jewson (2024): https://doi.org/10.5194/ascmo-11-1-2025.

More information and examples are available at https://www.fitdistcp.info, including the equivalent (more comprehensive) R package.

Development of this package was funded by the Lighthill Risk Network: https://lighthillrisknetwork.org.


### Example: Fitting a GEV distribution
```python
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import fitdistcp.genextreme

x = scipy.stats.genextreme.rvs(0, size=20)                  # make some example training data 
p = np.arange(0.001,0.999,0.001)                            # define the probabilities at which we wish to calculate the quantiles
q = fitdistcp.genextreme.ppf(x,p)                           # this command calculates two sets of predictive quantiles for the GEV, 
                                                            # one based on maxlik, and one that includes parameter uncertainty based on a calibrating prior
print(q['ml_params'])                                       # have a look at the maxlik parameters
plt.plot(q['ml_quantiles'],p, label='ML')                   # plot the maxlik quantiles
plt.plot(q['cp_quantiles'],p,color='red', label='CP')       # plot the quantiles that include parameter uncertainty
plt.legend()
plt.show()
```


### Models
The following models are currently supported. Let us know if you have any suggestions for other models to include.
- expon: Exponential distribution
- gamma: Gamma distribution
- genextreme: Generalised Extreme Value (GEV) distribution
- genextreme_p1 & genextreme_p12: Generalised Extreme Value (GEV) distribution, with 1 predictor, 2 predictors
- genpareto: Generalised Pareto distribution
- gumbel: Gumbel distribution
- lnorm: Lognormal distribution
- norm:	Normal distribution
- weibull:	Weibull distribution


### Methods
Four methods are provided for each model: ppf(x) (quantiles, 'percentage point function'), rvs(n, x) (random variates), pdf(x) and cdf(x), where x is the data to fit.
To use, e.g. ppf for the normal distribution, import fitdistcp.norm and call fitdistcp.norm.ppf(x).
To test the methods use e.g. fitdistcp.norm.reltest(). Reltests (reliability tests) are used to test the reliability of quantile calculations for making predictions.