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
from model import Model
from scipy.interpolate import interp1d



#%% class definition

class Model_intraday(Model):
    '''
    Class for point process model.
    '''

    def __init__(self, event_data, covariates, time_horizon):
        ''' Constructor of model
        event_data: data of point process (timestamps)
        covariates: exogenous data for completely observed factor, which need to
                    be interpolated (all data except for point process data)
        time_horizon: time horizon for point process data
        '''

        super().__init__(event_data, covariates, 2)
        
        self.time_horizon = time_horizon
        
        # interpolation function for covariate data
        self.interp_covariates = [interp1d(covar[:,0], covar[:,1],
                                           fill_value="extrapolate",
                                           kind="previous")
                                  for covar in covariates]
        
        self.break_points = np.sort(np.unique(np.hstack([covar[:,0] for covar in covariates])))
        
        # indices for parameters
        self.intensity_param_idx = [0,1,2]
        self.co_param_idx = []
        
        # indices of intensity parameters with zero bound
        self.zero_bound_idx_intensity = []
        self.zero_bound_idx_co = []
        
  
      
        
    def intensity_deterministic(self, intensity_param, co_param, time):
        ''' Specify intensity function
        intensity_param: parameters for shape of intensity function
        co_param: parameters for completely observed factor
        time: current time
        '''
        
        # intensity
        intensity = np.exp(intensity_param[0] +
                           intensity_param[1]*self.interp_covariates[0](time) +
                           intensity_param[2]*self.interp_covariates[1](time))

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
        out = np.sum(diff_bp*intensity_factors(self.break_points[:-1]))
        
        out -= time_horizon

        return out
