r"""
Data Analysis Library

Description:
This is a data analysis tool for basic statistics data analysis. 

Prerequisites:
matplotlib; pandas; scipy

Functionalities:
* Histogram plot; Box plot
* Descriptive statistics  
* Statistical estimations and inferences
* One-way ANOVA
* Correlation analysis; Linear regression
"""

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stat
from math import sqrt, ceil, log

Z_ALPHA = {0.005:2.5758, 0.025:1.96, 0.05:1.6449, 0.01:2.3263, 0.05:1.64485, 0.1:1.28155}


def histplot(data):
    """
    Histogram plot
    
    Input:
    data: a list of numbers
    
    Output:
    Histogram plot of the data
    
    Example:
    import random
    import statistics as s
    s.histplot(random.random(1000))
    """
    bins = ceil(1 + log(len(data))/0.693)
    plt.hist(data, bins)

 
def boxplot(data):
    """
    Box plot
    
    Input:
    data: a list of numbers

    Output:
    Box plot of the data

    Example:
    import random
    import statistics as s
    s.boxplot(random.random(1000))
    """  
    plt.box(data)
    
def mean(data):
    """
    Mean of an array
    """    
    return pd.Series(data).mean()

def var(data):
    """
    Variation of an array
    """
    return pd.Series(data).var()
    
def std(data):
    """
    Standard variation of an array
    """
    return pd.Series(data).std()

def corr(data1, data2):
	"""
	Correlation between two lists of data
	"""
	return pd.Series(data1).corr(pd.Series(data2))
	
def mean_est_std_known(data, alpha, std):
    """
    Confidence interval of the mean of single random variable
    * Standard variation is known.
    """
    if alpha/2.0 in Z_ALPHA:
        z = Z_ALPHA[alpha/2.0]
    else:
        z = stat.norm.ppf(1 - alpha/2.0)
    a = z * std / float(sqrt(len(data)))
    m = mean(data)
    conf_interval = [m - a, m + a]
    return conf_interval
    
def mean_est_std_unknown(data, alpha):
    """
    Confidence interval of the mean of single random variable
    * Standard variation is unknown.
    """
    len_data = len(data)
    tp = stat.t.ppf(1 - alpha/2.0, len_data-1)
    a = tp * std(data) / float(sqrt(len_data))
    m = mean(data)
    conf_interval = [m - a, m + a]
    return conf_interval

def mean_diff_est_std_known(data1, data2, std1, std2, alpha):
    """
    Confidence interval of the difference of two means of two random variables
    * Standard variations are known.
    """
    diff = abs(mean(data1) - mean(data2))
    if alpha/2.0 in Z_ALPHA:
        z = Z_ALPHA[alpha/2.0]
    else:
        z = stat.norm.ppf(1 - alpha/2.0)
    a = z * sqrt(std1/float(len(data1)) + std2/float(len(data2)))
    conf_interval = [diff - a, diff + a]
    return conf_interval

def mean_diff_est_std_equal_unknown(data1, data2, alpha):
    """
    Confidence interval of the difference of two means of two random variables
    * Standard variations are equal but unknown.
    """
    diff = abs(mean(data1) - mean(data2))
    len1, len2 = len(data1), len(data2)
    tp = stat.t.ppf(1 - alpha/2.0, len1 + len2 - 2)
    s = sqrt(((len1 - 1)*var(data1) + (len2 - 1)*var(data2)) / (len1 + len2 - 2))
    a = tp * s * sqrt(1.0/len1 + 1.0/len2)
    conf_interval = [diff - a, diff + a]
    return conf_interval

def mean_diff_est_std_unequal_unknown(data1, data2, alpha):
    """
    Confidence interval of the difference of two means of two random variables
    * Standard variations are unequal but unknown.
    """
    diff = abs(mean(data1) - mean(data2))
    len1, len2 = len(data1), len(data2)
    var1, var2 = var(data1), var(data2)
    df = (var1/len1 + var2/len2)*(var1/len1 + var2/len2) / ((var1*var1/len1/len1)/(len1-1) + (var2*var2/len2/len2)/(len2-1))
    tp = stat.t.ppf(1 - alpha/2.0, df)
    s = sqrt(((len1 - 1)*var(data1) + (len2 - 1)*var(data2)) / (len1 + len2 - 2))
    a = tp * s * sqrt(1.0/len1 + 1.0/len2)
    conf_interval = [diff - a, diff + a]
    return conf_interval

def var_est(data, alpha):
    """
    Confidence interval of the variation of single random variable
    """
    len_data = len(data)
    a = var(data) * (len_data - 1)
    conf_interval = [a / stat.chi2.ppf(1 - alpha/2.0, len_data - 1), a / stat.chi2.ppf(alpha/2.0, len_data - 1)]
    return conf_interval

def var_ratio_est(data1, data2, alpha):
    """
    Confidence interval of the ratio of two variations of two random variables
    """
    len1, len2 = len(data1), len(data2)
    a = var(data1) / var(data2)
    conf_interval = [a / stat.f.ppf(1 - alpha/2.0, len1 - 1, len2 - 1), a / stat.f.ppf(alpha/2.0, len1 - 1, len2 - 1)]
    return conf_interval

def mean_infer_doubleside_std_known(mu0, data, alpha, std):
    """
    Double side inference of the mean of single random variable
    * Standard variation is known.
    """
    if alpha/2.0 in Z_ALPHA:
        z = Z_ALPHA[alpha/2.0]
    else:
        z = stat.norm.ppf(1 - alpha/2.0)
    zscore = (mean(data) - mu0) * sqrt(len(data)) / std
    if zscore > -1*z and zscore < z: return True, zscore, z
    else: return False, zscore, z

def mean_infer_singleside_std_known(mu0, ge, data, alpha, std):
    """
    Single side inference of the mean of single random variable
    * Standard variation is known.
    """
    if alpha/2.0 in Z_ALPHA:
        z = Z_ALPHA[alpha/2.0]
    else:
        z = stat.norm.ppf(alpha)
    if ge is True: z *= -1.0
    zscore = (mean(data) - mu0) * sqrt(len(data)) / std
    if ge is True:
        if zscore > z: return True, zscore, z
        else: return False, zscore, z
    else:
        if zscore < z: return True, zscore, z
        else: return False, zscore, z
 
def mean_infer_doubleside_std_unknown(mu0, data, alpha):
    """
    Double side inference of the mean of single random variable
    * Standard variation is unknown.
    """
    len_data = len(data)
    tp = stat.t.ppf(1 - alpha/2.0, len_data-1)
    t = (mean(data)-mu0)*sqrt(len_data)/std(data)
    if t > -1*tp and t < tp: return True, t, tp
    else: return False, t, tp

def mean_infer_singleside_std_unknown(mu0, ge, data, alpha):
    """
    Single side inference of the mean of single random variable
    * Standard variation is unknown.
    """
    len_data = len(data)
    tp = stat.t.ppf(1 - alpha/2.0, len_data-1)
    if ge is True: tp *= -1.0
    t = (mean(data)-mu0)*sqrt(len_data)/std(data)
    if ge is True:
        if t > tp: return True, t, tp
        else: return False, t, tp
    else:
        if t < tp: return True, t, tp
        else: return False, t, tp

def var_infer_doubleside(sigma0, data, alpha):
    """
    Double side inference of the variation of single random variable
    """
    assert sigma0 != 0
    len_data = len(data)
    chi2 = var(data) * (len_data - 1) / sigma0 / sigma0
    left = stat.chi2.ppf(alpha/2.0, len_data - 1)
    right = stat.chi2.ppf(1 - alpha/2.0, len_data - 1)
    if chi2 > left and chi2 < right: return True, chi2, left, right
    else: return False, chi2, left, right

def var_infer_singleside(sigma0, ge, data, alpha):
    """
    Single side inference of the variation of single random variable
    """
    assert sigma0 != 0
    len_data = len(data)
    chi2 = var(data) * (len_data - 1) / sigma0 / sigma0
    left = stat.chi2.ppf(alpha/2.0, len_data - 1)
    right = stat.chi2.ppf(1 - alpha/2.0, len_data - 1)
    if ge is True:
        if chi2 > left: return True, chi2, left
        else: return False, chi2, left
    else:
        if chi2 < right: return True, chi2, right
        else: return False, chi2, right

def meandiff_infer_equal_std_known(data1, data2, std1, std2, alpha):
    """
    Double side inference of the difference of two means of two random variables
    * Standard variation is known
    """
    if alpha/2.0 in Z_ALPHA:
        z = Z_ALPHA[alpha/2.0]
    else:
        z = stat.norm.ppf(1 - alpha/2.0)
    zscore = (mean(data1) - mean(data2)) / sqrt(std1*std1/len(data1) + std2*std2/len(data2))
    if zscore > -1*z and zscore < z: return True, zscore, z
    else: return False, zscore, z

def meandiff_infer_ge_std_known(data1, data2, std1, std2, alpha, ge):
    """
    Single side inference of the difference of two means of two random variables
    * Standard variation is known
    """
    if alpha/2.0 in Z_ALPHA:
        z = Z_ALPHA[alpha/2.0]
    else:
        z = stat.norm.ppf(1 - alpha/2.0)
    zscore = (mean(data1) - mean(data2)) / sqrt(std1*std1/len(data1) + std2*std2/len(data2))
    if ge is True:
        if zscore > z: return True, zscore, z
        else: return False, zscore, z
    else:
        if zscore < -1*z: return True, zscore, -1*z
        else: return False, zscore, -1*z

def meandiff_infer_equal_std_unknown_equal(data1, data2, std1, std2, alpha):
    """
    Double side inference of the difference of two means of two random variables
    * Standard variation is equal but unknown
    """
    len1, len2 = len(data1), len(data2)
    t = stat.t.ppf(1 - alpha/2.0, len1+len2-2)
    s = sqrt(((len1 - 1)*var(data1) + (len2 - 1)*var(data2)) / (len1 + len2 - 2))
    tstat = (mean(data1) - mean(data2)) / s / sqrt(1.0/len(data1) + 1.0/len(data2))
    if tstat > -1*t and tstat < t: return True, tstat, t
    else: return False, tstat, t

def meandiff_infer_ge_std_unknown_equal(data1, data2, std1, std2, alpha, ge):
    """
    Single side inference of the difference of two means of two random variables
    * Standard variation is equal but unknown
    """
    len1, len2 = len(data1), len(data2)
    t = stat.t.ppf(1 - alpha/2.0, len1+len2-2)
    s = sqrt(((len1 - 1)*var(data1) + (len2 - 1)*var(data2)) / (len1 + len2 - 2))
    tstat = (mean(data1) - mean(data2)) / s / sqrt(1.0/len(data1) + 1.0/len(data2))
    if ge is True:
        if tstat > t: return True, tstat, t
        else: return False, tstat, t
    else:
        if tstat < -1*t: return True, tstat, -1*t
        else: return False, tstat, -1*t

def oneway_anova(data, alpha):
    """
    One-way ANOVA
    """
    h, w = len(data), len(data[0])
    num = h*w
    mean_col, mean_total = [0.0]*w, 0.0
    for i in xrange(w):
        for j in xrange(h):
            mean_col[i] += data[j][i]
    mean_total = sum(mean_col) / num
    for i in xrange(w):
        mean_col[i] /= h
    sst = 0.0
    for i in xrange(h):
        for j in xrange(w):
            sst += (data[i][j] - mean_total)*(data[i][j] - mean_total)
    sse = 0.0
    for i in xrange(w):
        for j in xrange(h):
            sse += (data[j][i] - mean_col[i])*(data[j][i] - mean_col[i])
    msa = (sst - sse) / (w - 1)
    mse = sse / (num - w)
    fstat = msa / mse
    f = stat.f.ppf(1 - alpha, w - 1, num - w)
    if fstat < f: return True, fstat, f
    else: return False, fstat, f

def is_corr(data1, data2, alpha):
	"""
	Descriptions:
	Hypothesis Test: If there is a linear relationship between two lists of data
	H0: There is no linear relationship; rho = 0
	H1: There is a linear relationship; rho != 0
	
	Inputs:
	data1: a list of numerical data
	data2: a list of numerical data
	alpha: significance level

	Outputs:
	Decision, t-statistic, critical value
	"""
	data_len = len(data1)
	assert data_len == len(data2), 'The length of two lists are not same.'
	rho = corr(data1, data2)
	t_stat = rho*sqrt(data_len-1) / sqrt(1-rho*rho)
	t = stat.t.ppf(1 - alpha/2.0, data_len-2)
	if tstat > -1*t and tstat < t: return True, tstat, t
	else: return False, tstat, t
    
def linear_regression(x, y, alpha):
	"""
	Descriptions:
	The correlation is tested in the first place.
	If there is no linear relationship, it is terminated.
	linear regression y = b0 + b1*x
	
	Input:
	x: a list of numbers 
	y: a list of numbers
	alpha: significance level
	
	Output:
	beta1: b1
	beta0: b0
	"""
	assert is_corr(x, y, 0.05)[0] is False, 'These is no linear relationship'
	sum_x, sum_y = sum(x), sum(y)
	len_data = len(x)
	sum_prod_xy = 0
	sum_xx = 0
	for i in xrange(len_data):
		sum_prod_xy += x[i]*y[i]
		sum_xx += x[i]*x[i]
	beta1 = (len_data*sum_prod_xy - sum_x*sum_y) / (len_data*sum_xx - sum_x)
	beta0 = sum_y/len_data - beta1*sum_x/len_data
	return beta1, beta0
