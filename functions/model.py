#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2020 by Anke Kramer

Licensed under the MIT License as stated in the LICENSE file.

This code supplements the paper "Exogenous factors for order arrivals 
on the intraday electricity market" by Anke Kramer and Rüdiger Kiesel.
"""

#%% class definition

class Model:
    '''
    Base class for point process model.
    '''

    def __init__(self, event_data, covariates, co_dim):
        ''' Constructor of model
        
        Inputs:
			event_data: data of point process (timestamps)
			covariates: exogenous data for completely observed factor (need to be interpolated)
			co_dim: dimension of completely observed factor data
        '''

        self.event_data = event_data
        self.covariates = covariates
        self.co_dim = co_dim


    def intensity_deterministic(self, intensity_param, co_state):
        ''' Specify intensity function
        
        Inputs:
			intensity_param: parameters for shape of intensity function
			co_state: state of completely observed factor
        '''

        return 0
