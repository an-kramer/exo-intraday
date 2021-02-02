#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2020 by Anke Kramer

Licensed under the MIT License as stated in the LICENSE file.

This code supplements the paper "Exogenous factors for order arrivals 
on the intraday electricity market" by Anke Kramer and Rüdiger Kiesel.
"""

#%% update Python path
import sys

sys.path.insert(0, '../functions')

#%% imports
from model_specs import Model_intraday as Model
from model_specs_noPV import Model_intraday as Model_noPV
from discretization import Discretization
import tools
import database as db
import optimization as optim
import warnings
import numpy as np
from multiprocessing import Pool


#%% script for executing algorithm

####################### set parameters and load data ##########################

# trading period and product
delivery_duration = 60
order_type = 'market_order'
ob_side = 'buy'

# load dates
dates_filename = 'examples'
toEstimate = tools.load_dates(dates_filename + '_toEstimate.csv')
done = tools.load_dates(dates_filename + '_' + ob_side + '_done.csv')
delivery_starts = [item for item in toEstimate if item not in done]



for num, delivery_start in enumerate(delivery_starts):
    # data import and processing
    
    print('ESTIMATING DATE ' + str(num+1) + '/' + str(len(delivery_starts)) + '...')
    
    try:
    
        # event data
        event_data, start, end, time_horizon = db.get_event_data(delivery_start=delivery_start,
                                                                 delivery_duration=delivery_duration,
                                                                 order_type=order_type,
                                                                 ob_side=ob_side,
                                                                 minutes_to_closure=180)
        
        # completely observed data (to be interpolated)
        co_data = [db.get_actuals_series(period_start=start,
                                         trading_end=end,
                                         actuals_type='solar',
                                         minutes_lag=15,
                                         unit='GWh',
                                         compute_error=True),
                   db.get_actuals_series(period_start=start,
                                         trading_end=end,
                                         actuals_type='wind',
                                         minutes_lag=15,
                                         unit='GWh',
                                         compute_error=True),
                   db.get_imbalance_data(period_start=start,
                                         trading_end=end,
                                         minutes_lag=15,
                                         unit="GWh")]

        
        # set names for building filenames
        model_title = 'FullModel'
        dataset = delivery_start.replace(':','').replace(' ','_')[:-2]
        
        # set flag for no PV (e.g. at night)
        if np.any(np.isnan(co_data[0][:,1])):
            no_pv = True
            co_data = co_data[1:]
        else:
            no_pv = False
        
        
    
        ##################### define model and discretization #########################
         # initialize model
        if no_pv:

            model = Model_noPV(event_data,
                               co_data,
                               time_horizon)
        else:

            model = Model(event_data,
                          co_data,
                          time_horizon)
            
        
        # initialize discretization
        disc = Discretization(model,
                              time_horizon)
        
        
        
        ########################## do optimization ####################################
        #define method for optimization
        method = 'Nelder-Mead'
        
        # reuse parameter estimates from self-exciting model
        param_old = tools.load_obj('../model_selfexciting/results/' + ob_side + '/ParamEstimation_SeModel_' + dataset + '.pkl', as_is=True)
        param_old = param_old['x']
        
        # depending on pv flag, set different start parameters
        if no_pv:
            
            # define starting values of model parameters
            param_0 = [np.concatenate((np.array([param_old[0]]), #alpha0
                                       np.array([10, 10]), # alpha2, alpha3
                                       param_old[1:])), #alpha4, beta
                       np.concatenate((np.array([param_old[0]]), #alpha0
                                       np.array([-0.4, -2]), # alpha2, alpha3
                                       param_old[1:])), #alpha4, beta
                       np.concatenate((np.array([param_old[0]]), #alpha0
                                       np.array([5, 5]), # alpha2, alpha3
                                       param_old[1:])), #alpha4 + beta
                       np.concatenate((np.array([param_old[0]]), #alpha0
                                       np.array([-0.5, -3]), # alpha2, alpha3
                                       param_old[1:]))] #alpha4, beta
            
            param_names = ['alpha0', 'alpha2', 'alpha3', 'alpha4', 'beta']

                
        else:
            
            # define starting values of model parameters
            param_0 = [np.concatenate((np.array([param_old[0]]), #alpha0
                                       np.array([10, 10, 10]), # alpha1, alpha2, alpha3
                                       param_old[1:])), #alpha4, beta
                       np.concatenate((np.array([param_old[0]]), #alpha0
                                       np.array([2.5, -0.4, -2]), # alpha1, alpha2, alpha3
                                       param_old[1:])), #alpha4, beta
                       np.concatenate((np.array([param_old[0]]), #alpha0
                                       np.array([-1, 5, 5]), # alpha1, alpha2, alpha3
                                       param_old[1:])), #alpha4, beta
                       np.concatenate((np.array([param_old[0]]), #alpha0
                                       np.array([-1, -0.5, -3]), # alpha1, alpha2, alpha3
                                       param_old[1:]))] #alpha4, beta
            
            param_names = ['alpha0', 'alpha1', 'alpha2', 'alpha3', 'alpha4', 'beta']
        
        
        # define function for optimizer (with only one vector as input)
        def estimate_param(x):
            return -disc.compute_likelihood(intensity_param=x[model.intensity_param_idx],
                                            co_param=np.array([x[model.co_param_idx]]))
            
            
        # start optimization
        def do_optimization_x0(param_0):
            return optim.do_optimization(objective_function=estimate_param,
                                         start_param=param_0,
                                         method='Nelder-Mead',
                                         callback=None,
                                         options={'maxfev': 1800},
                                         save=False,
                                         filename=ob_side + '/ParamEstimation_' + model_title + '_' + dataset,
                                         parameter_names=param_names)
        
        pool = Pool(4)    
        res = pool.map(do_optimization_x0, param_0)
        pool.close()
        pool.join()
        
        # choose best estimate
        res = tools.choose_best_estimate(res, save=True,
                                         filename=ob_side + '/ParamEstimation_' + model_title + '_' + dataset + '_notConverged')
        
        # get estimation result
        estimated = res['x'].copy()
        
        
        # standard errors
        se = disc.compute_standard_errors(params=estimated)
        res = tools.save_se(res, se, param_names, save=True, filename=ob_side + '/ParamEstimation_' + model_title + '_' + dataset)
        
        
        # add date to list if estimation was successful
        tools.add_date(date=delivery_start, date_list=done,
                       save=True, filename=dates_filename + '_' + ob_side + '_done.csv')
        
        
    except:

        warnings.warn("Exception raised in parameter estimation.")
        