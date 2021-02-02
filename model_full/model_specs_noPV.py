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
from model import Model
from scipy.interpolate import interp1d
import math


#%% class definition

class Model_intraday(Model):
    '''
    Class for point process model in Giesecke2018.
    '''

    def __init__(self, event_data, covariates, time_horizon):
        ''' Constructor of model
        event_data: data of point process (timestamps)
        covariates: exogenous data for completely observed factor, which need to
                    be interpolated (all data except for point process data)
        time_horizon: time horizon for point process data
        '''

        super().__init__(event_data, covariates, 3)
        
        self.time_horizon = time_horizon
        
        # interpolation function for covariate data
        self.interp_covariates = [interp1d(covar[:,0], covar[:,1],
                                           fill_value="extrapolate",
                                           kind="previous")
                                  for covar in covariates]
        
        self.break_points = np.sort(np.unique(np.hstack([covar[:,0] for covar in covariates])))
        
        # indices for parameters
        self.intensity_param_idx = [0,1,2,3]
        self.co_param_idx = [4]
        
        # indices of intensity parameters with zero bound
        self.zero_bound_idx_intensity = [3]
        self.zero_bound_idx_co = [0]


    
    def intensity_deterministic(self, intensity_param, co_param, time):
        ''' Specify intensity function
        intensity_param: parameters for shape of intensity function
        co_param: parameters for completely observed factor
        time: current time
        '''

        # index for Hawkes sum
        idx = bs.bisect_left(self.event_data[:,0],time)
        
        # intensity
        intensity = (np.exp(intensity_param[0] +
                            intensity_param[1]*self.interp_covariates[0](time) +
                            intensity_param[2]*self.interp_covariates[1](time)) +
                     intensity_param[3]*math.fsum(np.exp(-co_param[0]*(time-self.event_data[:idx,0]))))

        return intensity
    
    
    
    def loglik_integral(self, intensity_param, co_param, time_horizon):
        ''' Calculate the integral part of the log-likelihood function at
        least partly analytically.
        '''
        
        def intensity_factors(time):
            return np.exp(intensity_param[0] +
                          intensity_param[1]*self.interp_covariates[0](time) +
                          intensity_param[2]*self.interp_covariates[1](time))
        
        diff_bp = np.diff(self.break_points)
        val = intensity_factors(self.break_points[:-1])
        out = np.sum(diff_bp*val)
        
        out -= time_horizon
        
        out += (intensity_param[3]/co_param[0])*math.fsum(1-np.exp(-co_param[0]*(time_horizon-self.event_data[:,0])))

        return out
