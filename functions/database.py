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
import pandas as pd


#%% import event data
    
###############################################################################
def get_event_data(delivery_start, delivery_duration, order_type, ob_side,
                   minutes_to_closure=None, scale_par=None):
    '''
    Fetch event data and preprocess time series.
    
    Inputs:
		delivery_start: delivery start of product
		delivery_duration: in minutes, e.g. 60 for hourly contract
		order_type: 'market_order' or 'limit_order'
		ob_side: 'buy' or 'sell'
		minutes_to_closure: only specified time to gate closure is kept,
							otherwise whole trading period
		scale_par: scaling parameter
    '''
    
    # insert data import tailored for your data source here
    # the function must return a 1-column array with event times
    # the data from the abovementioned paper can be obtained from EPEX SPOT SE
    
    # data import for example data
    events = pd.read_csv('../data/events_{}.csv'.format(delivery_start.replace(' ', '_').replace(':','')[:-2])).values
    end = np.datetime64(delivery_start) - np.timedelta64(15,'m')
    start = end - np.timedelta64(minutes_to_closure,'m')
    time_horizon = minutes_to_closure
    
    return events, start, end, time_horizon
    


#%% import exogenous factors
    
###############################################################################
def get_actuals_series(period_start, trading_end,
                       actuals_type, minutes_lag=0,
                       unit=None, scale_par=None, compute_error=False):
    '''
    Fetch factors (solar, wind) from database.
    
    Inputs:
		period_start: start of data series (timestamp)
		trading_end: end of data series (timestamp)
		actuals_type: 'solar' or 'wind'
		minutes_lag: time lag in minutes
		unit: specify 'GWh', 'TWh' or None (for standard unit 'MWh')
		scale_par: scaling parameter
		compute_error: compute deviation of actuals from day-ahead forecast
    '''
    
    # insert data import tailored for your data source here
    # the function must return a 2-column array (first column: time, second column: values)
    # the data from the abovementioned paper can be obtained from Bundesnetzagentur | SMARD (smard.de/en)
    
    # data import for example data
    date = str(trading_end + np.timedelta64(15, 'm')).replace('T', '_').replace(':', '')[:-2]
    series = pd.read_csv('../data/{}_{}.csv'.format(actuals_type, date), header=None).values
    
    return series



###############################################################################
def get_imbalance_data(period_start, trading_end, minutes_lag, scale_par=None,
                       unit=None):
    ''' Fetch imbalance data (15 min. resolution) from database.
    
    Inputs:
        period_start: datetime at which the timeseries should start
        trading_end: datetime of trading end
        minutes_lag: time lag in minutes
        scale_par: scaling parameter
        unit: specify 'GWh', 'TWh' or None (for standard unit 'MWh')
    '''
    
    # insert data import tailored for your data source here
    # the function must return a 2-column array (first column: time, second column: values)
    # the data from the abovementioned paper can be obtained from Bundesnetzagentur | SMARD (smard.de/en)
    
    # data import for example data
    date = str(trading_end + np.timedelta64(15, 'm')).replace('T', '_').replace(':', '')[:-2]
    series = pd.read_csv('../data/balancing_{}.csv'.format(date), header=None).values
    
    return series
