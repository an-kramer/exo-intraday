#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2020 by Anke Kramer

Licensed under the MIT License as stated in the LICENSE file.

This code supplements the paper "Exogenous factors for order arrivals 
on the intraday electricity market" by Anke Kramer and Rüdiger Kiesel.
"""

#%% imports
import numpy as np
import bisect as bs
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import scipy.stats as stats


#%% functions

###############################################################################
def crop_data(data, time_horizon, left_part=True):
    '''
    Crop data.
    
    Inputs:
		data: 2-dim. array of data (first column contains timestamps)
		time_horizon: time at which data is cropped
    '''

    idx = bs.bisect_right(data[:,0], time_horizon)

    if left_part:
        return data[:idx,:]
    else:
        if time_horizon in data[:,0]:
            idx = idx-1
        return data[idx:,:]
    
    
    
###############################################################################
def save_estimates(optim_result,
                   param_names=None,
                   elapsed_time=None,
                   method=None,
                   start_param=None,
                   filename='estimation_result',
                   to_disc=True):
    ''' Update optimization result and save estimates of maximum likelihood
    approach.
    
    Inputs:
		optim_result: optimization result (dictionary)
		param_names: array of parameter names
		elapsed_time: elapsed time for parameter estimation
		method: method used in minimizer
		start_param: array of start parameters
		filename: filename for results
		to_disc: set to True for saving results to disc
    '''
    
    # build dictionary
    res = dict(optim_result)
    
    # add parameter specifications with names
    if param_names is not None:
        
        for idx,name in enumerate(param_names):
            res.update({'param_'+name:optim_result['x'][idx]})
    
    # add information on elapsed time in seconds
    if elapsed_time is not None:
        res.update({'elapsed_time (sec)':elapsed_time})
    
    # add information on estimation method
    if method is not None:
        res.update({'method':method})
        
    # add information on start parameters
    if start_param is not None:
        res.update({'start_param':start_param})       
    
    # save results to disc
    if to_disc:
        save_obj(res, filename)
        
    # return updated estimation result
    return res
    
    

###############################################################################
def save_obj(obj, name, as_is=False):
    ''' Save object with pickle.
    
    Inputs:
		obj: object
		name: filename
		as_is: set to True if 'name' specifies the complete saving path
    '''
    
    if as_is:
        with open(name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open('results/'+ name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        


###############################################################################
def load_obj(name, as_is=False):
    ''' Load pickled file.
    
    Inputs:
		name: filename
		as_is: set to True if 'name' specifies the complete saving path
    '''
    
    if as_is:
        with open(name, 'rb') as f:
            return pickle.load(f)
    else:
        with open('results/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)



###############################################################################
def generate_qq_plot(interarrival_times,
                     ci=True,
                     save=True,
                     show=False,
                     filename='qqplot'):
    ''' From given interarrival times generate the qq-plot with desired
    theoretical quantiles.
    
    Inputs:
		interarrival_times: array of interarrival times
		ci: set to True for computing confidence bands
		save: set to True for saving output
		show: set to True for showing output in command line
		filename: filename
    '''
        
    # get number of observations
    nobs = len(interarrival_times)
    
    # get confidence bands
    quantiles, upper, lower = conf_bands_exp(interarrival_times, 0.99)
    
    # generate qq-plot with 45-degree line
    fig, ax = plt.subplots(figsize=(3.2,4.8))
    
    ax.plot(quantiles, quantiles, 'r')
    
    # add confidence bands
    if ci:
        ax.plot(quantiles, upper, 'r--')
        ax.plot(quantiles, lower, 'r--')
    
    theoretical = stats.expon.ppf((np.arange(1.,nobs+1))/(nobs+1))
    ax.scatter(theoretical, np.sort(interarrival_times),
               facecolors='none', edgecolors='b', s=10, marker='8')
    ax.set_aspect('equal','box')
    
    
    # add labels
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    
    
    # save result
    if save:
        plt.savefig('plots/' + filename + '.png', dpi=300)
        
    # show plot
    if show:
        plt.show()
        
    # close figure
    plt.close()
    
        
        
###############################################################################
def conf_bands_exp(data, conf, method="pointwise"):
    ''' Compute the confidence bands for qq plot with data and theoretical
    exponential distribution.
    
    Inputs:
		data: data
		conf: confidence level
		method: "pointwise" computation of confidence bands
    '''
    
    # get sorted data sample
    data = np.sort(data)
    
    # get probabilities
    prob = (np.linspace(1,len(data), num=len(data))-0.5)/len(data)
    
    # theoretical quantiles
    theo_quant = stats.expon.ppf(prob)
    
    # reference line is diagonal (intercept 0, slope 1)
    a = 0
    b = 1
    ref_line = a+ b*theo_quant
    
    # get pointwise confidence bands
    z = stats.expon.ppf(prob)
    zz = stats.norm.ppf(1-(1-conf)/2)
    std_err = (b/stats.expon.pdf(z))*np.sqrt(prob*(1-prob)/len(data))
    
    # intervals
    upper = ref_line + zz*std_err
    lower = ref_line - zz*std_err
    
    return z, upper, lower



###############################################################################
def save_se(results, st_errors, param_names, save=False, filename=None):
    ''' Add standard errors of parameters to result set.
    
    Inputs:
		results: dictionary of estimation results
		st_errors: vector of standard errors
		param_names: list of parameter names
		save: set to True for saving results
		filename: set filename for saving
    '''
    
    # update dictionary with array of standard errors
    results.update({'standard_errors': st_errors})
    
    # update dictionary with single standard errors
    for err, name in zip(st_errors, param_names):
        description = 'param_' + name + '_se'
        results.update({description: err})
        
    # save results if desired
    if save:
        save_obj(results, filename)
        
    # return result set
    return results



###############################################################################
def load_dates(filename):
    ''' Load dates from filename.csv'''
    
    try:
        dates = pd.read_csv(filename, header=None)
        dates = dates.values
        return [str(date[0])[:19] for date in dates]
    
    except pd.errors.EmptyDataError:
        dates = []
        return dates



###############################################################################
def add_date(date, date_list, save=False, filename=None):
    ''' Add date to date list and save updated date list. '''
    
    date_list.append(date)
    
    # save new file
    if save:
        date_list = pd.DataFrame(date_list)
        date_list.to_csv(filename, sep=",", index=False, header=False)
        
    # return new list
    return [str(date[0])[:19] for date in date_list.values]



###############################################################################
def choose_best_estimate(results, save=False, filename=None):
    ''' Choose the result set with minimal negative likelihood function out of
    a list of result sets.
    
    Inputs:
		results: estimation restults
		save: set to True for saving unsuccessful estimations
		filename: filename
	'''
    
    # sort out None result
    results = [res for res in results if res is not None]
    
    # find minimum negativelikelihood
    min_loglik = min([res["fun"] for res in results if res["success"]])
    
    # find corresponding result set
    out = [res for res in results if res["fun"]==min_loglik][0]
    
    # check cases in which negative likelihood is smaller but algorithm did
    # not converge
    if min([res["fun"] for res in results]) != min_loglik:
        
        alt_result = [res for res in results if res["fun"]==min([res["fun"] for res in results])][0]
        
        if save:
            save_obj(alt_result, filename)
    
    #return output
    return out
