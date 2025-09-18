import fitdistcp.genextreme
import numpy as np
import scipy.stats
import warnings
import pyextremes
import pandas as pd
from pyextremes import get_model

warnings.filterwarnings('ignore')


# Parameters
ntrials = 1000    # we're going to test prediction methods over 1000 trials  
nsample = 30      # for a sample size of 30
xi_0 = 0.1        # 'true' shape parameter
p = 0.99          # probability that defines the quantile of interest
rp0 = 1/(1-p)     # convert p to a return period

print("Goal: for a perfectly reliable prediction, we should get rp=100")

sum_ML = 0
sum_CP = 0
    
for i in range(ntrials):
    # Make the GEV distributed training data
    x = scipy.stats.genextreme.rvs(c=-xi_0, loc=0, scale=1, size=nsample)

    # use pyextremes MLE method to calculate quantile
    x_pandas = pd.Series(x, index=pd.date_range('1/1/2025', periods=nsample, freq='365.2425D'))
    model = pyextremes.EVA(x_pandas)
    model.get_extremes(method='BM', block_size='365.2425D')
    model.fit_model(distribution = scipy.stats.genextreme)
    q_ml = model.get_summary(return_period=[rp0]).values[0][0]

    # use fitdistcp to calculate quantile
    q_cp = fitdistcp.genextreme.ppf(x, p)['cp_quantiles']
   
    # Compare the predictions against truth
    sum_ML += 1 - scipy.stats.genextreme.cdf(q_ml, c=-xi_0, loc=0, scale=1)
    sum_CP += 1 - scipy.stats.genextreme.cdf(q_cp, c=-xi_0, loc=0, scale=1)


# Average over the trials
PCP_ML = sum_ML / ntrials
PCP_CP = sum_CP / ntrials

# Calculate associated return periods
rp_ML = 1 / PCP_ML
rp_CP = 1 / PCP_CP

# Print predicted RPs
print(f"maxlik: rp={rp_ML}")  
print(f"calibrating prior: rp={rp_CP}")    

print("Conclusion: the maxlik and Lmoments predictions underestimate the return period.")
print("Calibrating prior prediction also underestimates, but by much less.")

# journal paper: https://ascmo.copernicus.org/articles/11/1/2025/
# fitdistcp software library: www.fitdistcp.info, and on CRAN