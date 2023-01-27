import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Cambria"
import numpy as np
import xlsxwriter as xlw
import os
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

#create a dataset
data = {'Product':["A_promo","A_promo","A_std","A_std"],'Conversion':[0,0,1,1]}
data = pd.DataFrame(data)

data.info()

# state hypotheses & set acceptance criteria
# null_hypothesis = "There is no relationship between mailer type and signup rate. They are independent."
# alternate_hypothesis = "There is a relationship between mailer type and signup rate. They are not independent."
acceptance_criteria = 0.05

promo_results = data[data['Product'] == 'A_promo']['Conversion']
standard_results = data[data['Product'] == 'A_std']['Conversion']

n_con = promo_results.count()
n_treat = standard_results.count()
successes = [promo_results.sum(), standard_results.sum()]
nobs = [n_con, n_treat]

z_stat, pval = proportions_ztest(successes, nobs=nobs)
(lower_con, lower_treat), (upper_con, upper_treat) = proportion_confint(successes, nobs=nobs, alpha=0.05)

if pval <= acceptance_criteria:
    print("The p-value is at or below the acceptance criteria. You can reject the null hypothesis.")
    print(f'z statistic: {z_stat:.2f}')
    print(f'p-value: {pval:.3f}')
    print(f'ci 95% for control group: [{lower_con:.3f}, {upper_con:.3f}]')
    print(f'ci 95% for treatment group: [{lower_treat:.3f}, {upper_treat:.3f}]')
    
else:
    print("The p-value is above the acceptance criteria. You can not reject the null hypothesis, you have to accept the null hypothesis.")
    print(f'z statistic: {z_stat:.2f}')
    print(f'p-value: {pval:.3f}')
    print(f'ci 95% for control group: [{lower_con:.3f}, {upper_con:.3f}]')
    print(f'ci 95% for treatment group: [{lower_treat:.3f}, {upper_treat:.3f}]')
    