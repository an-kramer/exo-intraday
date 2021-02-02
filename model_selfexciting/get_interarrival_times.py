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
import tools
import database as db

import numpy as np
import os
import pandas as pd
import scipy as sp


#%% script for computation of interarrival times

# set inital parameters
ob_side = 'buy'
path = 'results/' + ob_side
order_type = 'market_order'
delivery_duration = 60



####################### load filenames, set model parameters ##########################

# load filenames
files = os.listdir(path)
files = [file for file in files if file.endswith('.pkl')
         and not '_notConverged' in file]
files.sort()



####################### load saving variable ##########################

# check if file with interarrival times exists

if os.path.exists(os.path.join(os.getcwd(), 'results/' + ob_side + '/interarrival_times/interarrival_times.pkl')):
    
    itimes_dict = tools.load_obj(os.path.join(os.getcwd(), 'results/' + ob_side + '/interarrival_times/interarrival_times.pkl'),
                                 as_is=True)
    
else:
    
    # check if directory exists already, otherwise create it
    if not os.path.isdir(os.path.join(os.getcwd(), 'results/' + ob_side + '/interarrival_times')):
        
        os.mkdir(os.path.join(os.getcwd(), 'results/' + ob_side + '/interarrival_times'))
        
    # initialize dictionary for interarrival times
    itimes_dict = {'timestamp_UTC': (), 'interarrival_times': ()}




for num, file in enumerate(files):
    # data import and processing
    
    print('PROCESSING DATE ' + str(num+1) + '/' + str(len(files)) + '...')
    
        
    # load result
    res = tools.load_obj('results/' + ob_side + '/' + file, as_is=True)
    estimated = res['x']
    param_names = [item.replace('param_','').replace('_se','')
                   for item in res.keys() if item.startswith('param')
                   and not item.endswith('se')]
    delivery_start = str(pd.to_datetime(file[-19:-4], format='%Y-%m-%d_%H%M'))
    
    if res['success'] and not delivery_start in itimes_dict["timestamp_UTC"]:
    
        # event data
        event_data, start, end, time_horizon = db.get_event_data(delivery_start=delivery_start,
                                                                 delivery_duration=delivery_duration,
                                                                 order_type=order_type,
                                                                 ob_side=ob_side,
                                                                 minutes_to_closure=180)
        
        
        co_data = None
    
        
    
        ##################### define model and discretization #########################
        # initialize model
        model = Model(event_data,
                      co_data,
                      time_horizon)

        
        
        ##################### calculate interarrival times ############################
        # get filtered path
        def intensity(time):
            return model.intensity_deterministic(intensity_param=estimated[model.intensity_param_idx],
                                                 co_param=np.array([estimated[model.co_param_idx]]),
                                                 time=time)
        
        times = event_data[:,0]
        interarrival_times = np.array([sp.integrate.quad(intensity,lbound,rbound, points=model.break_points) 
                                       for lbound, rbound in zip(times[:-1], times[1:])])
        

                        
        
        ##################### save results ############################
        if not len(interarrival_times) == 0:
            itimes_dict["timestamp_UTC"] = itimes_dict["timestamp_UTC"] + (delivery_start,)
            itimes_dict["interarrival_times"] = itimes_dict["interarrival_times"] + (interarrival_times[:,0],)
            tools.save_obj(itimes_dict, name='results/' + ob_side + '/interarrival_times/interarrival_times.pkl', as_is=True)
