# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 15:15:56 2021

@author: vragu
Model Portfolio Optimization when returns have a skew
"""

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def convert_to_alpha(s):
    d=(np.pi/2*((abs(s)**(2/3))/(abs(s)**(2/3)+((4-np.pi)/2)**(2/3))))**0.5
    if s<0:
        d *= -1
    a=((d)/((1-d**2)**.5))
    return(a)

def createSkewDist(mean, sd, skew, size):

    if abs(skew) >= 0.99:
        # calculate the degrees of freedom 1 required to obtain the specific skewness statistic, derived from simulations
        loglog_slope=-2.211897875506251 
        loglog_intercept=1.002555437670879 
        df2=500
        df1 = 10**(loglog_slope*np.log10(abs(skew)) + loglog_intercept)

        # sample from F distribution
        fsample = np.sort(stats.f(df1, df2).rvs(size=size))

        # adjust the variance by scaling the distance from each point to the distribution mean by a constant, derived from simulations
        k1_slope = 0.5670830069364579
        k1_intercept = -0.09239985798819927
        k2_slope = 0.5823114978219056
        k2_intercept = -0.11748300123471256

        scaling_slope = abs(skew)*k1_slope + k1_intercept
        scaling_intercept = abs(skew)*k2_slope + k2_intercept

        scale_factor = (sd - scaling_intercept)/scaling_slope    
        new_dist = (fsample - np.mean(fsample))*scale_factor + fsample

        # flip the distribution if specified skew is negative
        if skew < 0:
            new_dist = np.mean(new_dist) - new_dist

        # adjust the distribution mean to the specified value
        final_dist = new_dist + (mean - np.mean(new_dist))
    
    elif skew != 0:
        # Use skewnormal distribution
        location, scale, shape = skewnorm_params(mean, sd, skew)
        final_dist = stats.skewnorm.rvs(shape,loc=location, scale = scale, size=10000)
    
    else:
        # if skew = 0, just use the normal distribution 
        final_dist = np.random.normal(mean, sd, size)
        
    return final_dist


# Function above does not really work for low levels of skews, so here is another one that uses skewnormal() distribution
# Based on these formulas: https://en.wikipedia.org/wiki/Skew_normal_distribution
from math import sqrt, pi
def skewnorm_moments(location, scale, shape):
    ''' Calcualtes moments of a skew normal distribution from parameters'''
    
    alpha = shape #Alias used in math books
    delta = alpha/sqrt(1+alpha**2)

    mean = location + scale * delta * sqrt(2/pi)
    var  = scale ** 2 * (1 - 2 * delta **2 / pi)
    std  = sqrt(var)
    skew = (4-pi) / 2 * (delta * sqrt(2/pi))**3 / (1 - 2 * delta**2 / pi)**(3/2)
    
    return mean, std, skew

# Inverse to skewnorm_moments.  From desired moments, solves for skewnormal distribution parameters
def skewnorm_params(mean, std, skew):
    ''' Calculates parameters of the skew normal distribution from moments '''
    if abs(skew) >= 1.0:
        print("Error - can't solve for skewness >= 1.")
    else:  
        # Solve for shape (alpha)
        alpha = convert_to_alpha(skew)
        d = alpha / sqrt(1+alpha**2)
        shape = alpha
        
        # Solve for scale:
        scale = sqrt(std**2 / (1 - 2* d**2/pi))
        
        # Solve for location
        location = mean - scale * d * sqrt(2/pi)
        
    return location, scale, shape

''' Testing Skew Normal Calculations '''
#Set seeds
if __name__ == "__main__":
    np.random.seed(1)

    test1 = False
    if test1:
        params = skewnorm_params(1,3,-0.5)
        print(params)
        moments = skewnorm_moments(*params)
        print(moments)
    
    # '''EXAMPLE - testing that my function is producing correct skews'''
    desired_mean = 0.06
    desired_skew = -1.2
    desired_sd = 0.16
    
    final_dist = createSkewDist(mean=desired_mean, sd=desired_sd, skew=desired_skew, size=1000000)
    
    # inspect the plots & moments, try random sample
    fig, ax = plt.subplots(figsize=(12,7))
    sns.histplot(final_dist, ax=ax, color='green', label='generated distribution', stat='density')
    sns.histplot(np.random.choice(final_dist, size=10000), ax=ax, color='red', line_kws={'alpha':.1}, 
                  stat='density', label='sample n=1000')
    ax.legend()
    
    print('Input mean: ', desired_mean)
    print('Result mean: ', np.mean(final_dist),'\n')
    
    print('Input SD: ', desired_sd)
    print('Result SD: ', np.std(final_dist),'\n')
    
    print('Input skew: ', desired_skew)
    print('Result skew: ', stats.skew(final_dist))
