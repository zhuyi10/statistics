r"""
Data Analysis Library

Descriptions:
This is a data analysis tool for basic statistics data analysis. 

Prerequisites:
matplotlib; pandas; scipy

Functionalities:
* Histogram plot; Box plot
* Descriptive statistics  
* Statistical estimations and inferences
* One-way ANOVA
* Correlation analysis; Linear regression
* Multi-variable statistical estimations and inferences
* Discriminant analysis
"""

import sys
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stat
import math
from math import sqrt, ceil, log, exp

# Critical values for z distribution.
# The key is alpha. The significant level is 1 - alpha.
# The value is z-statistics, stat.norm.ppf(1 - alpha)
Z_ALPHA = {0.005:2.5758, 0.025:1.96, 0.05:1.6449, 0.01:2.3263, 0.05:1.64485, 0.1:1.28155}

# Critical values for t distribution.
# The key is alpha. The significant level is 1 - alpha.
# The value is dictionary, in which
# the key is the degree of freedom, which is 19 < df < 40;
# the value is t-statistics, stat.t.ppf(1 - alpha, degree_freedom) 
T_ALPHA = {0.005:[], 0.025:[], 0.05:[], 0.01:[], 0.05:[], 0.1:[]}
for k, v in T_ALPHA.items():
    v = [stat.t.ppf(1 - k, i) for i in xrange(19, 41)]


def create_discrete_distribution(values, probabilities):
    """
    Create a possibility distribution of discrete random variable.
    
    Input:
    values: a list of ordered values of a discrete random variable
    probabilities: a list of probabilities of the ordered values
    
    Output:
    dist: a dictionary that represents the probability density distribution
    
    Example:
    import random
    values = range(10)
    probabilities = [random.random() for i in range(10)]
    sp = sum(probabilities)
    probabilities = [p/sp for p in probabilities]
    dist = create_discrete_distribution(values, probabilities)
    """
    assert len(values) == len(probabilities), 'The lengths of values dose not match the lengths of probabilities.'
    dist = dict()
    for v, p in zip(values, probabilities):
        dist[v] = p
    return dist


def plot_hist(data, show=True):
    """
    Plot histogram.

    Args:
        data: a list of numbers
        show: if True, show the plot; if False, return the data
    Returns:
        hist_data or None
    Example:
        import random
        import statistics as s
        data = s.histogram([random.random for i in xrange(1000)])
        s.plot_hist(data)
    """
    hist_data = plt.hist(data)
    if show:
        plt.show()
    else:
        return hist_data


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
    
    Input:
    data: a list of numbers

    Output:
    Mean of the data
    
    Example:
    import random
    import statistics as s
    data = random.random(1000)
    print s.mean(data)
    """    
    return pd.Series(data).mean()


def var(data):
    """
    Variation of an array

    Input:
    data: a list of numbers

    Output:
    Variation of the data
    
    Example:
    import random
    import statistics as s
    data = random.random(1000)
    print s.var(data)
    """    
    return pd.Series(data).var()

    
def std(data):
    """
    Standard variation of an array
    
    Input:
    data: a list of numbers

    Output:
    Standard variation of the data
    
    Example:
    import random
    import statistics as s
    data = random.random(1000)
    print s.std(data)
    """
    return pd.Series(data).std()


def corr(data1, data2):
    """
    Correlation between two lists of data

    Input:
    data1: a list of numbers
    data2: a list of numbers

    Output:
    Correlation between data1 & data2
    
    Example:
    import random
    import statistics as s
    data1 = random.random(1000)
    data2 = random.random(1000)
    print s.corr(data1, data2)
    """
    return pd.Series(data1).corr(pd.Series(data2))

    
def mean_est_std_known(data, alpha, std):
    """
    Confidence interval of the mean of single random variable
    * The distribution of the data follows normal distribution.
    * Otherwise, the sample size len(data) > 30.
    * Standard variation is known.
    
    Input:
    data: a list of numbers
    alpha: significant level
    std: the standard variation of the data

    Output:
    The confidence interval
    
    Example:
    import random
    import statistics as s
    data = random.random(1000)
    print str(mean_est_std_known(data, 0.05, 1.0))
    """
    a = Z_ALPHA.get(alpha/2.0, stat.norm.ppf(1 - alpha/2.0)) * std / float(sqrt(len(data)))
    m = mean(data)
    conf_interval = [m - a, m + a]
    return conf_interval

    
def mean_est_std_unknown(data, alpha):
    """
    Confidence interval of the mean of single random variable
    * The distribution of the data follows normal distribution.
    * Standard variation is unknown.
    
    Input:
    data: a list of numbers
    alpha: the significant level

    Output:
    The confidence interval
    
    Example:
    import random
    import statistics as s
    data = random.random(1000)
    print str(mean_est_std_unknown(data, 0.05))
    """
    len_data = len(data)
    sl, df = 1 - alpha/2.0, len_data - 1
    if sl in T_ALPHA:
        if df in T_ALPHA[sl]:
            tp = T_ALPHA[sl][df]
    else:
        tp = stat.t.ppf(sl, df)
    a = tp * std(data) / float(sqrt(len_data))
    m = mean(data)
    conf_interval = [m - a, m + a]
    return conf_interval


def mean_diff_est_std_known(data1, data2, std1, std2, alpha):
    """
    Confidence interval of the difference of two means of two random variables
    * Two random variables are independent.
    * The distribution of the data follows normal distribution.
    * Otherwise, the sample size len(data) > 30.
    * Standard variations are known.
    
    Input:
    data1: a list of numbers
    data2: a list of numbers
    alpha: significant level
    std1: the standard variation of data1
    std2: the standard variation of data1

    Output:
    The confidence interval
    
    Example:
    import random
    import statistics as s
    data1 = random.random(1000)
    data2 = random.random(1000)
    print str(mean_diff_est_std_known(data1, data2, 1.0, 1.0, 0.05))
    """
    diff = abs(mean(data1) - mean(data2))
    a = Z_ALPHA.get(alpha/2.0, stat.norm.ppf(1 - alpha/2.0)) * sqrt(std1/float(len(data1)) + std2/float(len(data2)))
    conf_interval = [diff - a, diff + a]
    return conf_interval


def mean_diff_est_std_equal_unknown(data1, data2, alpha):
    """
    Confidence interval of the difference of two means of two random variables
    * Two random variables are independent.
    * The distribution of the data follows normal distribution.
    * Standard variations are equal but unknown.
    
    Input:
    data1: a list of numbers
    data2: a list of numbers
    alpha: significant level

    Output:
    The confidence interval
    
    Example:
    import random
    import statistics as s
    data1 = random.random(1000)
    data2 = random.random(1000)
    print str(mean_diff_est_std_equal_unknown(data1, data2, 0.05))
    """
    diff = abs(mean(data1) - mean(data2))
    len1, len2 = len(data1), len(data2)
    sl, df = 1 - alpha/2.0, len1 + len2 - 2
    if sl in T_ALPHA:
        if df in T_ALPHA[sl]:
            tp = T_ALPHA[sl][df]
    else:
        tp = stat.t.ppf(sl, df)
    s = sqrt(((len1 - 1)*var(data1) + (len2 - 1)*var(data2)) / (len1 + len2 - 2))
    a = tp * s * sqrt(1.0/len1 + 1.0/len2)
    conf_interval = [diff - a, diff + a]
    return conf_interval


def mean_diff_est_std_unequal_unknown(data1, data2, alpha):
    """
    Confidence interval of the difference of two means of two random variables
    * Two random variables are independent.
    * The distribution of the data follows normal distribution.
    * Standard variations are unequal and unknown.
    
    Input:
    data1: a list of numbers
    data2: a list of numbers
    alpha: significant level

    Output:
    The confidence interval
    
    Example:
    import random
    import statistics as s
    data1 = random.random(1000)
    data2 = random.random(1000)
    print str(mean_diff_est_std_unequal_unknown(data1, data2, 0.05))
    """
    diff = abs(mean(data1) - mean(data2))
    len1, len2 = len(data1), len(data2)
    var1, var2 = var(data1), var(data2)
    sl = 1 - alpha/2.0
    df = (var1/len1 + var2/len2)*(var1/len1 + var2/len2) / ((var1*var1/len1/len1)/(len1-1) + (var2*var2/len2/len2)/(len2-1))
    if sl in T_ALPHA:
        if df in T_ALPHA[sl]:
            tp = T_ALPHA[sl][df]
    else:
        tp = stat.t.ppf(sl, df)
    s = sqrt(((len1 - 1)*var(data1) + (len2 - 1)*var(data2)) / (len1 + len2 - 2))
    a = tp * s * sqrt(1.0/len1 + 1.0/len2)
    conf_interval = [diff - a, diff + a]
    return conf_interval


def var_est(data, alpha):
    """
    Confidence interval of the variation of single random variable

    Input:
    data: a list of numbers
    alpha: significant level

    Output:
    conf_interval: confidence interval
    
    Example:
    import random
    import statistics as s
    data = random.random(1000)
    print str(var_est(data, 0.05))
    """
    len_data = len(data)
    a = var(data) * (len_data - 1)
    conf_interval = [a / stat.chi2.ppf(1 - alpha/2.0, len_data - 1), a / stat.chi2.ppf(alpha/2.0, len_data - 1)]
    return conf_interval


def var_ratio_est(data1, data2, alpha):
    """
    Confidence interval of the ratio of two variations of two random variables

    Input:
    data1: a list of numbers
    data2: a list of numbers
    alpha: significant level

    Output:
    conf_interval: confidence interval
    
    Example:
    import random
    import statistics as s
    data1 = random.random(1000)
    data2 = [random.random()+1 for i in xrange(1000)]
    print str(var_ratio_est(data1, data2, 0.05))
    """
    len1, len2 = len(data1), len(data2)
    a = var(data1) / var(data2)
    conf_interval = [a / stat.f.ppf(1 - alpha/2.0, len1 - 1, len2 - 1), a / stat.f.ppf(alpha/2.0, len1 - 1, len2 - 1)]
    return conf_interval


def mean_infer_doubleside_std_known(mu0, data, alpha, std):
    """
    Double side inference of the mean of single random variable
    * Standard variation is known.

    Input:
    mu0: infer if the sample data is from a distribution with mean equals to mu0
    data: a sample data
    alpha: significant level
    std: standard variation

    Output:
    decision: Ture, if it equals to mu0; otherwise False

    Example:
    import random
    import statistics as s
    data = random.random(1000)
    print mean_infer_doubleside_std_known(1.0, data, 0.05, 1)
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

    Input:
    mu0: infer if the sample data is from a distribution with mean greater/smaller than mu0
    ge: True means greater; False means smaller
    data: a sample data
    alpha: significant level
    std: standard variation

    Output:
    decision: the hypothesis is Ture/False

    Example:
    import random
    import statistics as s
    data = random.random(1000)
    print mean_infer_doubleside_std_known(1.0, True, data, 0.05, 1)
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
    r2: r square value
    f_stat: f statistics
    f: critical value
    """
    assert is_corr(x, y, 0.05)[0] is False, 'These is no linear relationship'
    # solve linear regression
    sum_x, sum_y = sum(x), sum(y)
    len_data = len(x)
    sum_prod_xy = 0
    sum_xx = 0
    for i in xrange(len_data):
        sum_prod_xy += x[i]*y[i]
        sum_xx += x[i]*x[i]
    beta1 = (len_data*sum_prod_xy - sum_x*sum_y) / (len_data*sum_xx - sum_x)
    beta0 = sum_y/len_data - beta1*sum_x/len_data
    # evaluate linear regression
    est_y = [x0*beta1 + beta0 for x0 in x]
    mean_y = mean(y)
    sst = sum([(y[i] - mean_y) * (y[i] - mean_y) for i in xrange(len_data)])
    ssr = sum([(est_y[i] - mean_y) * (est_y[i] - mean_y) for i in xrange(len_data)])
    sst = sum([(y[i] - est_y[i]) * (y[i] - est_y[i]) for i in xrange(len_data)])
    r2 = ssr / sst
    f_stat = ssr * (len_data - 2) / sse
    f = stat.f.ppf(1 - alpha, 1, len_data - 2)
    return beta1, beta0, r2, f_stat, f

def multi_mean(data):
    """
    The vector of means of multiple random variables
    
    Input:
    data: n-by-p matrix contains n samples with p features
    
    Output:
    1-by-p vector of means
    """
    return np.mean(data, axis=0)

def multi_cov(data):
    """
    Covariance matrix of multiple random variables
    
    Input:
    data: n-by-p matrix contains n samples with p features
    
    Output:
    p-by-p covariance matrix
    """    
    return np.cov(data, rowvar=0)
    
def multi_corr(data):
    """
    Correlation coefficients matrix of multiple random variables
    
    Input:
    data: n-by-p matrix contains n samples with p features
    
    Output:
    p-by-p correlation coefficients matrix
    """    
    c = np.cov(data, rowvar=0)
    h, w = len(c), len(c[0])
    for i in xrange(h):
        r.append = []
        cii = c[i][i]
        for j in xrange(w):
            r[i].append(c[i][j]/sqrt(cii)/sqrt(c[j][j]))
    return r

def single_multi_meandiff_infer_cov_know(data, cov, mu, alpha):
    """
    Double side inference of the mean of a data set

    * The data set is from multi-variable normal distribution.
    * Covariance matrix is known.
    
    Input:
    data: n-by-p matrix contains n samples with p features
    
    Output:
    decision: True means the mean equals to mu.
    chi_stat: chi-square statistic
    c_val: critical value
    """
    n, p = float(len(data)), float(len(data[0]))
    mean_diff = multi_mean(data) - np.array(mu)
    chi_stat = n*la.dot(la.dot(np.transpose(diff_mean), la.inv(cov)), diff_mean)
    c_val = stat.chi2.ppf(alpha, p)
    if chi_stat < c_val:
        return True, chi_stat, c_val
    else:
        return False, chi_stat, c_val

def single_multi_meandiff_infer_cov_unknow(data, mu, alpha):
    """
    Double side inference of the mean of a data set

    * The data set is from multi-variable normal distribution.
    * Covariance matrix is unknown.
    
    Input:
    data: n-by-p matrix contains n samples with p features
    mu: In H0, the mean of a data set equals to mu
    alpha: significant level
    
    Output:
    decision: True means the mean equals to mu.
    chi_stat: chi-square statistic
    c_val: critical value
    """
    n, p = float(len(data)), float(len(data[0]))
    mean_diff = multi_mean(data) - np.array(mu)
    f_stat = (n-p)/p*la.dot(la.dot(sqrt(n)*np.transpose(diff_mean), la.inv(multi_cov)), sqrt(n)*diff_mean)
    c_val = stat.f.ppf(alpha, p, n-p)
    if f_stat < c_val:
        return True, f_stat, c_val
    else:
        return False, f_stat, c_val

def double_multi_meandiff_infer_equal_cov_known(data1, data2, cov, alpha):
    """
    Double side inference of the two means of two data sets

    * Two data sets are from two multi-variable normal distributions
    * Covariance matrix are known and identical
    
    Input:
    data1: n-by-p matrix contains n samples with p features
    data2: m-by-p matrix contains n samples with p features
    cov: the covariance matrix
    alpha: significant level
    
    Output:
    decision: True means the two means are the same.
    chi_stat: chi-square statistic
    c_val: critical value
    """        
    diff_mean = multi_mean(data1) - multi_mean(data2)
    n, m, p = float(len(data1)), float(len(data2)), float(len(data1[0]))
    chi_stat = n*m/(n+m)*la.dot(la.dot(np.transpose(diff_mean), la.inv(cov)), diff_mean)
    c_val = stat.chi2.ppf(alpha, p)
    if chi_stat < c_val:
        return True, chi_stat, c_val
    else:
        return False, chi_stat, c_val
        
def double_multi_meandiff_infer_equal_cov_unknown(data1, data2, alpha):
    """
    Double side inference of the two means of two data sets

    * Two data sets are from two multi-variable normal distributions
    * Covariance matrix are unknown and identical
    
    Input:
    data1: n-by-p matrix contains n samples with p features
    data2: m-by-p matrix contains n samples with p features
    alpha: significant level

    Output:
    decision: True means the two means are the same.
    f_stat: chi-square statistic
    c_val: critical value
    """        
    diff_mean = multi_mean(data1) - multi_mean(data2)
    n, m, p = float(len(data1)), float(len(data2)), float(len(data1[0]))
    a = sqrt(n*m/(n+m))
    f_stat = (n+m-p-1)/p*la.dot(a*la.dot(np.transpose(diff_mean), la.inv(multi_cov(data1)+multi_cov(data2))), a*diff_mean)
    c_val = stat.f.ppf(alpha, p, n+m-p-1)
    if f_stat < c_val:
        return True, f_stat, c_val
    else:
        return False, f_stat, c_val

def double_multi_meandiff_infer_diff_cov_equal_size(data1, data2, alpha):
    """
    Double side inference of the two means of two data sets

    * Two data sets are from two multi-variable normal distributions.
    * Covariance matrix are unknown and different.
    * The size of data sets are the same.
    
    Input:
    data1: n-by-p matrix contains n samples with p features
    data2: n-by-p matrix contains n samples with p features
    alpha: significant level
    
    Output:
    decision: True means the two means are the same.
    f_stat: chi-square statistic
    c_val: critical value
    """        
    n, p = float(len(data1)), float(len(data1[0]))
    diff_mean = multi_mean(data1) - multi_mean(data2)
    z = []
    for i in xrange(n):
        z.append([data1[i][j]-data2[i][j] for j in xrange(p)])
    z = np.array(z)
    s = np.array()
    for i in xrange(n):
        s = la.dot(z[i]-diff_mean, np.transpose(z[i]-diff_mean))
    f_stat = (n-p)*n/p*la.dot(la.dot(np.transpose(diff_mean), la.inv(s)), diff_mean)
    c_val = stat.f.ppf(alpha, p, n-p)
    if f_stat < c_val:
        return True, f_stat, c_val
    else:
        return False, f_stat, c_val
        
def double_multi_meandiff_infer_diff_cov_diff_size(data1, data2, alpha):
    """
    Double side inference of the two means of two data sets

    * Two data sets are from two multi-variable normal distributions.
    * Covariance matrix are unknown and different.
    * The size of data1 is smaller than that of data2.
    
    Input:
    data1: n-by-p matrix contains n samples with p features
    data2: m-by-p matrix contains n samples with p features
    alpha: significant level

    Output:
    decision: True means the two means are the same.
    f_stat: chi-square statistic
    c_val: critical value
    """        
    n, m, p = len(data1), len(data2), len(data1[0])
    diff_mean = multi_mean(data1) - multi_mean(data2)
    z = []
    a, b = [0]*n, [0]*m
    for i in xrange(n):
        for j in xrange(p):
            a[j] += data2[i][j]
    c = 1/sqrt(n*m)
    a = [i*c for i in a]
    for i in xrange(m):
        for j in xrange(p):
            b[j] += data2[i][j]
    c = 1/m
    b = [i*c for i in b]
    for i in xrange(n):
        z.append([data1[i][j]-sqrt(n/m)*data2[i][j]+a-b for j in xrange(p)])
    z = np.array(z)
    s = np.array()
    for i in xrange(n):
        s = la.dot(z[i]-diff_mean, np.transpose(z[i]-diff_mean))
    f_stat = (n-p)*n/p*la.dot(la.dot(np.transpose(diff_mean), la.inv(s)), diff_mean)
    c_val = stat.f.ppf(alpha, p, n-p)
    if f_stat < c_val:
        return True, f_stat, c_val
    else:
        return False, f_stat, c_val

def multi_independent_infer(data, alpha):
    """
    Independent inference of multiple random variables

    * The data follows p-dimensional normal distribution.
    
    Input:
    data: n-by-p matrix contains n samples with p features
    alpha: significant level

    Output:
    decision: True means they are independent
    chi_stat: chi-square statistic
    c_val: critical value
    """        
    n, p, mean_data = len(data), len(data[0]), multi_mean(data)
    s = np.zeros((p, p))
    for i in xrange(n):
        s += np.dot(data[i]-mean_data, np.transpose(data[i]-mean_data))
    chi_stat = -2*log(exp(-1.5*np.trace(s))*pow(np.linalg.det(s), n/2)*pow(math.e/n, n*p/2))
    c_val = stat.chi2.ppf(alpha, p*(p+1)/2)
    if chi_stat < c_val:
        return True, chi_stat, c_val
    else:
        return False, chi_stat, c_val
        
def multi_cov_infer(data, sigma, alpha):
    """
    Covariance matrix inference of multiple random variables

    * The data follows p-dimensional normal distribution.
    
    Input:
    data: n-by-p matrix contains n samples with p features
    sigma: a covariance matrix
    alpha: significant level

    Output:
    decision: True means the data's covariance matrix equals to sigma
    chi_stat: chi-square statistic
    c_val: critical value
    """        
    n, p = len(data), len(data[0])
    d = la.inv(la.cholesky(data))
    y = []
    for i in xrange(n):
        y.append(la.dot(d, a[i]))
    return multi_independent_infer(y, alpha)

    
def mahalanobis_dist(u, v, c, sqrt=True):
    """
    Mahalanobis distance between two vectors.
    
    Input:
    u: a vector
    v: a vector
    c: covariance matrix
    
    Output:
    dist: Mahalanobis distance
    """
    u = np.array(u)
    v = np.array(v)
    c = np.array(c)
    diff = u - v
    m = np.dot(np.dot(diff, c), diff)
    return np.sqrt(m) if sqrt else m


def bayesian_discriminant(data, means, priors, costs, cum_funcs):
    """
    Bayesian discriminant analysis

    Input:
    data: a sample data to be discriminanted
    means: list of means of each class
    prior: list of prior probabilities of each class
    costs: matrix of costs of failures
    cum_funcs: list of cumulative probability functions of each class

    Output:
    The class this data belongs to
    """
    rs = 0
    num_class = len(priors)
    assert num_class == len(costs) and num_class == len(cum_funcs),
           'The length of input lists dose not equal to the number of class'
    min_cost = sys.float_info.max
    for i in xrange(num_class):
        tmp_cost = 0
        for j in xrange(num_class):
            if means[j] < means[i]:
                tmp_cost += priors[j] * costs[i][j] * (1 - cum_funcs[j](data))
            else:
                tmp_cost += priors[j] * costs[i][j] * cum_funcs[j](data)
        if tmp_cost < min_cost:
            min_cost = tmp_cost
            rs = i
    return rs


def fisher_discriminant_vector(data, means, covs):
    """
    Fisher discriminant analysis

    Input:
    data: a sample data to be discriminanted
    means: list of means of each class
    covs: list of covariance matrixes

    Output:
    The vector used for discriminant
    """
    num_p = len(data)
    assert num_p == len(means[0]) and num_p == len(covs[0][0])
    sum_cov = np.sum(np.array(covs), 0)
    inv_sum_cov = la.inv(sum_cov)
    mt = numpy.eye(num_p) - (1 / num_p) * numpy.ones((num_p, num_p))
    mb = np.dot((np.dot(np.transpose(np.array(means)), mt), np.array(means))
    eig_val, eig_vec = numpy.linalg.eig(np.dot(inv_sum_cov, mb))
    max_val = sys.float_info.min
    max_ind = 0
    for i, v in enumerate(eig_val):
        if v > max_val:
            max_val = v
            max_ind = i
    return eig_vec[:, max_ind]
