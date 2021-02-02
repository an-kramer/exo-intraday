#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2020 by Anke Kramer

Licensed under the MIT License as stated in the LICENSE file.

This code supplements the paper "Exogenous factors for order arrivals 
on the intraday electricity market" by Anke Kramer and Rüdiger Kiesel.
"""

#%% imports
from model import Model
import tools
import numpy as np
import numdifftools as nd
import scipy.integrate as integrate


#%% class definition

class Discretization:
    '''
    Base class for discretization.
    '''

    def __init__(self, model, time_horizon, timesteps=1000):
        ''' Constructor of discretization
        model: model
        time_horizon: time horizon of data (e.g. 180 minutes)
        timesteps: number of discrete timesteps until time horizon,
                    might for example be used for calculating intensity
                    at certain points
        '''

        if not isinstance(model, Model):
            raise ValueError('model is not of class Model.')

        self.model = model
        self.time_horizon = time_horizon


        # crop data at time horizon
        self.model.event_data = tools.crop_data(self.model.event_data,
                                                self.time_horizon)

		# generate equidistant sequence
        equidist_times = np.linspace(0,time_horizon, timesteps)
        self.time_vec = equidist_times

    
    
    
    def event_likelihood_deterministic(self,
                                       intensity_param,
                                       track_intensity=False,
                                       co_param=None,
                                       compute_se=False):
        ''' Compute event likelihood of a model.
        
        Inputs:
			intensity_param: parameter vector for intensity
			track_intensity: bool; indicates whether path of intensity is calculated
			co_param: parameter vector for completely observed factor
			compute_se: set to True if standard errors shall be computed
        '''
        
        # check parameter bounds
        if (intensity_param[self.model.zero_bound_idx_intensity]<0).any():
            print('Parameter required to be nonnegative.')
            return -np.inf
        
        if co_param is not None:
            if (co_param[self.model.zero_bound_idx_co]<0).any():
                print('Parameter required to be nonnegative.')
                return -np.inf
        
        # define intensity function only depending on time
        def intensity_here(time):
            return self.model.intensity_deterministic(intensity_param=intensity_param,
                                                      co_param=co_param,
                                                      time=time)-1
        
        if track_intensity:
            
            return np.array([intensity_here(time)+1 for time in self.time_vec])
            
        else:
            
            # initialize likelihood value
            loglik = 0
            
            # compute first part with logarithm
            loglik += np.sum(np.log(np.array([intensity_here(time)+1 
                                              for time in self.model.event_data[:,0]])))
                                                                   
            # compute second part with integral
            if hasattr(self.model, 'loglik_integral'):
                integral = self.model.loglik_integral(intensity_param, co_param, self.time_horizon)
            else:
                integral = integrate.quad(intensity_here, 0, self.time_horizon, points=self.model.break_points)[0]
            
            loglik -= integral
            
            # return output
            return loglik




    # function for computing the likelihood
    def compute_likelihood(self,
                           intensity_param,
                           co_param):
        '''
        Compute the overall log likelihood (sum of event log likelihood and
        factor log likelihood) for the given model.

		Inputs:
			intensity_param: parameter set for specified intensity function
			co_param: parameter set for completely observed factor
        '''

        return self.event_likelihood_deterministic(intensity_param,
                                                   co_param=co_param)
    
    
    
    
    # function for computing the path of the intensity function
    def tracking_intensity(self,
                           intensity_param,
                           co_param):
        '''
        Compute the path of the intensity.
        
        Inputs:
			intensity_param: parameter vector for intensity parameters
			co_param: parameter vector for completely observed factor
        '''
        
        # compute intensity path           
        intensity_path = self.event_likelihood_deterministic(intensity_param=intensity_param,
                                                             track_intensity=True,
                                                             co_param=co_param)
        
        # return intensity
        return intensity_path
    
    
    
        
    # function for computing standard errors
    def compute_standard_errors(self, params):
        '''
        Compute the standard errors of parameter estimates.
        
        Inputs:
			params: vector of parameters
        '''
                   
        def loglik(x):
            return self.event_likelihood_deterministic(intensity_param=x[self.model.intensity_param_idx],
                                                       co_param=x[self.model.co_param_idx])
        
        # finite time horizon approximation, see online appendix of Azizpour et al. (2018)
        var_covar = np.linalg.inv(-1/self.time_horizon*nd.Hessian(loglik, method='forward')(params))
    
        
        # return the standard errors (diagonal elements)
        # standard error of mean: sqrt(diagonal elements)/sqrt(sample size)
        return np.sqrt(np.diagonal(var_covar))/np.sqrt(self.time_horizon)
