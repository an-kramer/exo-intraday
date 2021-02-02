#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2020 by Anke Kramer

Licensed under the MIT License as stated in the LICENSE file.

This code supplements the paper "Exogenous factors for order arrivals 
on the intraday electricity market" by Anke Kramer and Rüdiger Kiesel.
"""

#%% imports
import tools
from scipy.optimize import minimize
import time


#%% functions

###############################################################################
def do_optimization(objective_function,
                    start_param,
                    method='Nelder-Mead',
                    options={},
                    callback=None, 
                    save=False,
                    filename=None,
                    parameter_names=None):
    ''' Minimize objective function and return/save results.
    
    Inputs:
        objective_function: objective function
        start_param: array with start parameters
        method: method for optimization
        options: options for optimization, e.g. number of function evaluations
        callback: lambda function for callback in optimization
        save: set to True for saving results to disc
        filename: filename for saving results
        parameter_names: list of parameter names in model
    '''
    
    # start optimization
    start_time = time.time()
    
    try:
        res = minimize(objective_function,
                       start_param,
                       method=method,
                       options=options,
                       callback=callback)
        
    except:
        res = None
            
    end_time = time.time()
    elapsed_time = end_time-start_time
    
    # save results if desired
    if save:
        to_disc = True
    else:
        to_disc = False
    
    # update results
    if res is not None:    
        res = tools.save_estimates(optim_result=res,
                                   param_names=parameter_names,
                                   elapsed_time=elapsed_time,
                                   method=method,
                                   start_param=start_param,
                                   filename=filename,
                                   to_disc=to_disc)
        
    # return estimation result
    return res
