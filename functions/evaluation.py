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
import tools

import os
from scipy.stats import norm, chi2, expon, wasserstein_distance, anderson
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np
import re
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import pandas as pd


#%% plot parameter estimates

###############################################################################
def plot_estimates(path, alpha=95, save=False, saving_path=None, show=False,
                   param_names=None, hour_list=None):
    ''' Plot parameter estimates with confidence intervals based on standard
    errors and significance levels.
    
    Input:
        path: path of result files
        alphas: significance level for parameter confidence intervals
        save: set to True for saving graphics
        saving_path: path where graphics are saved
        show: set to True for saving graphics
        param_names: prespecified list of parameter names (needed if a parameter might appear only at certain times)
        hour_list: list of hours to be evaluated
    '''
    
    # get list of files
    files = os.listdir(path)
    files = [file for file in files if file.endswith('.pkl')
                                    and not '_notConverged' in file]
    files.sort()
    
    # loop through files and gather data
    stats = get_estimates_from_files(path, files, param_names)
    
    # generate errorbar plot
    if hour_list is None:
        plot_errorbars(statistics=stats, alpha=alpha, save=save,
                       saving_path=saving_path)
    else:
        plot_errorbars_hourly(statistics=stats, hour_list=hour_list,
                              alpha=alpha, save=save, saving_path=saving_path)
    
    
    
###############################################################################
def get_estimates_from_files(path, files, param_names=None):
    ''' Collect all respective parameter estimates and standard errors from
    the specified files.
    
    Input:
        path: path of result files
        files: list of filenames
        param_names: prespecified list of parameter names (needed if a parameter might appear only at certain times)
    '''
    
    # preallocate output list
    
    if param_names is None:
        results = tools.load_obj(path + '/' + files[0], as_is=True)
        param_names = [item.replace('param_','').replace('_se','')
                       for item in results.keys() if item.startswith('param')
                       and item.endswith('se')]
    
    out = [{'parameter': [], 'standard_error': [], 'date': [], 'success': []}.copy()
           for item in range(len(param_names))]
    
    # add parameter names to output
    for num, name in enumerate(param_names):
        out[num].update({'param_name': name})
    
    # loop through files
    for file in files:
        
            results = tools.load_obj(path + '/' + file, as_is=True)

            match = re.search(r'\d{4}-\d{2}-\d{2}_\d{4}', file)
            date = pd.to_datetime(datetime.strptime(match.group(), '%Y-%m-%d_%H%M'), utc=True)
            success = results['success']
            
            for num, par in enumerate(param_names):
                out[num]['date'].append(date)
                out[num]['success'].append(success)
                try:
                    out[num]['parameter'].append(results['param_' + par])
                    out[num]['standard_error'].append(results['param_' + par + '_se'])
                except KeyError:
                    out[num]['parameter'].append(np.nan)
                    out[num]['standard_error'].append(np.nan)
            
    # convert lists to numpy arrays
    for item in out:
        item['parameter'] = np.array(item['parameter'])
        item['standard_error'] = np.array(item['standard_error'])
        
    # return output
    return out



###############################################################################
def plot_errorbars(statistics, alpha=95, save=False, saving_path=None, show=False):
    ''' Generate plot of parameters with respective confidence intervals.
    
    Inputs:
        statistics: list of result dictionaries
        alpha: significance level for parameter confidence intervals
        save: set to True fo saving figures
        saving_path: set saving path for figures
        show: set to True for showing figures
    '''
    
    matplotlib.rcParams.update({'font.size': 12})
    
    # calculate z-score
    z_score = norm.ppf(1-(1-alpha/100)/2)
    
    # set date format
    date_fmt = mdates.DateFormatter('%Y-%m-%d')
    
    # loop through all parameters
    for stats in statistics:
        
        upper = stats['parameter'] + stats['standard_error']*z_score
        lower = stats['parameter'] - stats['standard_error']*z_score
        
        upper_q = np.nanquantile(upper, .99)
        lower_q = np.nanquantile(lower, .01)
        
        fig, ax = plt.subplots(figsize=(20,5))
        
        ax.axhline(y=0, color='red', alpha=0.6)
        
        ax.errorbar(mdates.date2num(stats['date']), stats['parameter'],
                    yerr=stats['standard_error']*z_score,
                    fmt='.k', ecolor='darkgrey', capsize=3)
        
        ax.set_ylim((lower_q, upper_q))
        
        ax.grid(alpha=0.5, linestyle=':')
        
        ax.xaxis.set_major_formatter(date_fmt)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title('Parameter ' + stats['param_name'])
        
        # save figure if desired
        if save:
            plt.savefig(saving_path + '/' + stats['param_name'] + '_alpha' + str(alpha) + '.png', dpi=300)
        
        # show figure if desired
        if show:
            plt.show()
            
        # close figures
        plt.close()
        
        
        
###############################################################################
def plot_errorbars_hourly(statistics, hour_list, alpha=95,
                          save=False, saving_path=None, show=False):
    ''' Generate plot of parameters with respective confidence intervals.
    
    Inputs:
        statistics: list of result dictionaries
        alpha: significance level for parameter confidence intervals
        hour_list: list of hours to be evaluated
        save: set to True fo saving figures
        saving_path: set saving path for figures
        show: set to True for showing figures
    '''
    
    matplotlib.rcParams.update({'font.size': 22})
    
    # calculate z-score
    z_score = norm.ppf(1-(1-alpha/100)/2)
    
    # set date format
    date_fmt = mdates.DateFormatter('%Y-%m-%d')
    
    # loop through all parameters
    for stats in statistics:
        
        # convert dates to local time
        stats['date'] = [date.tz_convert('Europe/Berlin') for date in stats['date']]
        
        for hour in hour_list:
            
            # get statistics only for desired hour
            idx_temp = [num for num, date in enumerate(stats['date']) if date.hour==hour]
            stats_temp = stats.copy()
            stats_temp['date'] = np.array(stats_temp['date'])[idx_temp].tolist()
            stats_temp['parameter'] = stats_temp['parameter'][idx_temp]
            stats_temp['standard_error'] = stats_temp['standard_error'][idx_temp]
            stats_temp['success'] = np.array(stats_temp['success'])[idx_temp].tolist()
        
            if not np.all(np.isnan(stats_temp['parameter'])):
                upper = stats_temp['parameter'] + stats_temp['standard_error']*z_score
                lower = stats_temp['parameter'] - stats_temp['standard_error']*z_score
                
                upper_q = np.nanquantile(upper, .99)
                lower_q = np.nanquantile(lower, .01)
                
                fig, ax = plt.subplots(figsize=(20,5))
                
                ax.axhline(y=0, color='red', alpha=0.6)
                
                ax.errorbar(mdates.date2num(stats_temp['date']), stats_temp['parameter'],
                            yerr=stats_temp['standard_error']*z_score,
                            fmt='.k', ecolor='darkgrey', capsize=3)
                
                ax.set_ylim((lower_q, upper_q))
                
                ax.grid(alpha=0.5, linestyle=':')
                
                ax.xaxis.set_major_formatter(date_fmt)
                
                ax.set_xlabel('Date')
                ax.set_ylabel('Value')
                #ax.set_title('Parameter ' + stats['param_name'])
                
                plt.tight_layout()
                
                # save figure if desired
                if save:
                    plt.savefig(saving_path + '/' + stats_temp['param_name'] + '_alpha' + str(alpha) + '_hour' + str(hour) + '.png', dpi=300)
                
                # show figure if desired
                if show:
                    plt.show()
                    
            # close figures
            plt.close()
        



#%% Likelihood ratio test for nested models
    
###############################################################################
def do_lr_statistics(path_full, path_nested, confidence_lvl=[90, 95, 99],
                     save=False, saving_path=None, model_names=None,
                     hour_list=None, plot=False):
    ''' Generate a table and evaluate the likelihood ratio test for all
    dates.
    
    Inputs:
        path_full: path of full model
        path_nested: path of nested model
        confidence_lvl: list of confidence levels
        save: set to True for saving results
        saving_path: saving path
        model_names: list of model names (full + nested model)
        hour_list: list of hours for wich statistics are computed
        plot: set to True for plotting results
    '''
    
    # check model name input
    if model_names is not None:
        if len(model_names) != 2:
            raise ValueError('Please specify two model names.')
            
    # get results of the likelihood ratio test
    lr_results = do_lr_test(path_full, path_nested)
    
    # gather statistics
    if hour_list is None:
        
        num_models = lr_results.shape[0]
        avg_p = np.mean(lr_results['p-value'].values)
        
        num_significant = np.zeros(len(confidence_lvl))
        for num, lvl in enumerate(confidence_lvl):
            num_significant[num] = np.sum(lr_results['p-value'].values < np.round((100-lvl)/100,4))
            
        df = pd.DataFrame()
        df['Models'] = [int(num_models)]
        df['Avg. p-val'] = [round(avg_p,3)]
        
        for lvl, num in zip(confidence_lvl, num_significant):
            df['p < ' + str((100-lvl)/100)] = [int(num)]
            
    else:
        
        df = pd.DataFrame(columns=[str(hour) for hour in hour_list],
                          index=["Models", "Avg. p-val"]+['p < ' + str(np.round((100-lvl)/100,4)) for lvl in confidence_lvl])
        lr_results.index = lr_results.index.tz_localize('UTC').tz_convert('Europe/Berlin')
        
        for count, hour in enumerate(hour_list):
            
            # get data frame index for current hour and temporary dataframe
            idx_temp = [date for date in lr_results.index if date.hour==hour]
            lr_results_temp = lr_results.loc[idx_temp]
            
            if len(lr_results_temp) == 0:
                df[str(count)] = [0, 0] + [0 for lvl in confidence_lvl]
            else:
                num_models = lr_results_temp.shape[0]
                avg_p = np.mean(lr_results_temp['p-value'].values)
                
                num_significant = np.zeros(len(confidence_lvl))
                for num, lvl in enumerate(confidence_lvl):
                    num_significant[num] = np.sum(lr_results_temp['p-value'].values < (100-lvl)/100)
                    
                df[str(count)] = [int(num_models), round(avg_p,3)] + list(num_significant)
                
        if plot:
            stacked_bars_lr(df, save=save, saving_path=saving_path)
    
    # save results
    if save:
        if hour_list is None:
            idx = False
        else:
            idx=True
            
        if model_names is None:
            df.to_csv(saving_path + '/LR_result.csv', sep=';',
                      index=idx)
        else:
            df.to_csv(saving_path + '/LR_result_' + model_names[0] + '_' + model_names[1] + '.csv', sep=';',
                      index=idx)
            
    # return output
    return df
        
    
    

###############################################################################
def do_lr_test(path_full, path_nested):
    ''' Wrapper function for doing likelihood ratio test for all estimates.
    
    Input:
        path_full: path of results for full model
        path_nested: path of results for nested model
    '''
    
    # get list of files for full model
    files_full = os.listdir(path_full)
    files_full = [file for file in files_full if file.endswith('.pkl')
                                              and not '_notConverged' in file]
    
    # get list of files for nested model
    files_nested = os.listdir(path_nested)
    files_nested = [file for file in files_nested if file.endswith('.pkl')
                                                  and not '_notConverged' in file]
    
    # get list of matched dates
    files, dates = match_files(files1=files_full, files2=files_nested)
    
    # loop through matched list and compute likelihood ratio test
    test_results = pd.DataFrame(index=dates, columns=['LR', 'p-value'])
    for idx, item in zip(dates, files):
        
        # get results
        res_full = tools.load_obj(path_full + '/' + item[0], as_is=True)
        res_nested = tools.load_obj(path_nested + '/' + item[1], as_is=True)
        
        if res_full['success'] and res_nested['success']:
        
            # get number of parameters and compute degrees of freedom for test
            params_full = len([key.replace('param_','').replace('_se','')
                               for key in res_full.keys() if key.startswith('param')
                               and key.endswith('se')])
            params_nested = len([key.replace('param_','').replace('_se','')
                               for key in res_nested.keys() if key.startswith('param')
                               and key.endswith('se')])
            dof = params_full - params_nested
            
            # get likelihood values
            loglik_full = -res_full['fun']
            loglik_nested = -res_nested['fun']
            
            # conduct LR test
            LR, pval = lr_test(loglik_full, loglik_nested, dof)
            
            # write results to dataframe
            test_results.loc[idx,'LR'] = LR
            test_results.loc[idx,'p-value'] = pval
            
        else:
            
            test_results = test_results.drop(idx)
        
    # return output
    return test_results
    
    
    
###############################################################################
def match_files(files1, files2):
    ''' Get two lists of filenames and match them according to the date in the
    filename.
    
    Input:
        files1: list of files
        files2: list of files
    '''
    
    # get all dates from both file lists
    dates1 = {item[-19:-4] for item in files1}
    dates2 = {item[-19:-4] for item in files2}
    
    # get prefixes of models
    prefix1 = files1[0][:-19]
    prefix2 = files2[0][:-19]
    
    # get suffix
    suffix = '.pkl'
    
    # find matching dates
    date_matches = sorted(list(dates1.intersection(dates2)))
    
    # build matches with complete filenames
    matches = [(prefix1 + d + suffix, prefix2 + d + suffix)
               for d in date_matches]
    
    # get list of dates separately
    dates = [pd.to_datetime(d.replace('_',' ')) for d in date_matches]
            
    # return output
    return matches, dates
        
    
    
###############################################################################
def lr_test(loglik_full, loglik_nested, dof):
    ''' Conduct likelihood ratio test.
    
    Input:
        loglik_full: maximized log-likelihood value for full (larger) model
        loglik_nested: maximized log-likelihood value for nested (smaller) model
        dof: degrees of freedom, corresponds to difference in number of parameters
    '''
    
    # compute test statistic
    LR = 2*(loglik_full - loglik_nested)
    
    # compute p-value
    pval = chi2.sf(LR, dof)
    
    # return output
    return LR, pval



###############################################################################
def stacked_bars_lr(dataframe, save=False, saving_path=None):
    ''' Barplot for likelihood ratio test results.
    
    Inputs:
        dataframe: dataframe with likelihood ratio test results
        save: set to True for saving figures
        saving_path: path for saving figures
    '''
    
    # get row numbers of dataframe
    row_num = dataframe.shape[0]
    
    # get integers for hours
    hours = [int(item) for item in dataframe.columns]
    
    # predefine colorlists
    col_red = ['firebrick', 'indianred', 'salmon']
    col_blue = ['darkblue', 'steelblue', 'skyblue']
    
    # plot stacked bars
    matplotlib.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots()
    
    ax.bar(hours, dataframe.iloc[-1], color=col_red[0], label=dataframe.index[-1])
    
    if row_num > 3:
        todo = reversed(range(2,row_num-1))
        for row, col in zip(todo, col_red[1:]):
            ax.bar(hours, dataframe.iloc[row]-dataframe.iloc[row+1],
                   bottom=dataframe.iloc[row+1], color=col,
                   label=dataframe.index[row])
            
    # plot reference bars with model number
    ax.bar(hours, dataframe.iloc[0]-dataframe.iloc[2],
           bottom=dataframe.iloc[2], color='lightgray', label='Total models')


    # labels
    ax.set_xlabel('Hour')
    ax.set_ylabel('Number of models', color=col_red[0])
    ax.tick_params(axis='y', labelcolor=col_red[0])
    
    
    # second axis with percentage values
    ax2 = ax.twinx()
    ax2.set_ylabel('Percentage', color=col_blue[0])
    ax2.tick_params(axis='y', labelcolor=col_blue[0])
    
    ax2.plot(hours, (dataframe.iloc[2]/dataframe.iloc[0])*100, color=col_blue[0],
             linewidth=3, label=dataframe.index[2])
    ax2.set_ylim((0,100))
    
    if row_num > 3:
        todo = range(3,row_num)
        for row, col in zip(todo, col_blue[1:]):
            ax2.plot(hours, (dataframe.iloc[row]/dataframe.iloc[0])*100,
                     color=col, linewidth=3, label=dataframe.index[row])
            
            
    # custom legend
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    
    handles_all = []
    labels_all = []
    
    for label, handle in zip(labels1, handles1):
        labels_all.append(label)
        handle_match = [hand for lab,hand in zip(labels2,handles2)
                        if lab==label]
        if len(handle_match) > 0:
            handles_all.append((handle,handle_match[0]))
        else:
            handles_all.append((handle,))
    
    fig.legend(tuple(handles_all), tuple(labels_all), loc="lower left",
               bbox_to_anchor=(0,0.92,1,0), mode="expand", ncol=row_num-1)
    
    if save:
        plt.savefig(saving_path + '/lr_barplot.png', dpi=300,
                    bbox_inches="tight")
        


#%% interarrival times

###############################################################################
def compare_dist_to_exp(path_full, path_nested, model_names=None, hour_list=None,
                        plot_outliers=False, save=False, saving_path=None):
    ''' Compare distance of time-changed interarrival times to exponential
    distribution.
    
    Inputs:
        path_full: path of full model results
        path_nested: pth of nested model results
        model_names: list of model names
        hour_list: list of hours if hourly evaluation is desired
        plot_outliers: set to True for plotting outliers in boxplot
        save: set to True for saving graphics
        saving_path: specify path for saving graphics
    '''
    
    # check model name input
    if model_names is not None:
        if len(model_names) != 2:
            raise ValueError('Please specify two model names.')
    
    # get interarrival times for both models
    itimes_full = tools.load_obj(path_full + '/interarrival_times.pkl', as_is=True)
    itimes_nested = tools.load_obj(path_nested + '/interarrival_times.pkl', as_is=True)
    
    if hour_list is None:
        # get matched interarrival times
        itimes1, itimes2 = match_interarrival_times(itimes_full, itimes_nested)
        
        # compute distances to exponential distribution
        dist1 = []
        dist2 = []
        
        for t1, t2 in zip(itimes1, itimes2):
            
            dist1.append(dist_to_exp(t1))
            dist2.append(dist_to_exp(t2))
            
        # plot result
        plot_boxplots(itimes_list=[dist1,dist2], model_names=model_names,
                      plot_outliers=plot_outliers, save=save, saving_path=saving_path)
        
    else:
        hour_dict={}
        for hour in hour_list:
            
            # get matched interarrival times
            itimes1, itimes2 = match_interarrival_times(itimes_full, itimes_nested, hour=hour)
            
            # compute distances to exponential distribution
            dist1 = []
            dist2 = []
            
            for t1, t2 in zip(itimes1, itimes2):
                
                dist1.append(dist_to_exp(t1))
                dist2.append(dist_to_exp(t2))
                
            hour_dict.update({str(hour): [dist1, dist2]})
                
        # plot result
        plot_boxplots(itimes_list=hour_dict, model_names=model_names,
                      hour_list=hour_list, plot_outliers=plot_outliers,
                      save=save, saving_path=saving_path)
        
    # return output
    return dist1, dist2
        
   

###############################################################################
def match_interarrival_times(times1, times2, hour=None):
    ''' Match dates for interarrival times times1 and times2.
    If hour-specific matching is desired, specify parameter 'hour'.
    '''
    
    # get all dates from both file lists
    if hour is None:
        dates1 = times1["timestamp_UTC"]
        dates2 = times2["timestamp_UTC"]
    else:
        dates1 = pd.to_datetime(times1["timestamp_UTC"], utc=True).tz_convert('Europe/Berlin')
        dates2 = pd.to_datetime(times2["timestamp_UTC"], utc=True).tz_convert('Europe/Berlin')
        
        dates1 = [str(date.tz_convert('UTC'))[:19] for date in dates1 if date.hour==hour]
        dates2 = [str(date.tz_convert('UTC'))[:19] for date in dates2 if date.hour==hour]
    
    # find matching dates
    date_matches = sorted(list(set(dates1).intersection(set(dates2))))
    
    # get interarrival times for both models
    itimes1 = [item for item, date in zip(times1["interarrival_times"], times1["timestamp_UTC"]) if date in date_matches]
    itimes2 = [item for item, date in zip(times2["interarrival_times"], times2["timestamp_UTC"]) if date in date_matches]
    
    # collect indices where interarrival time is infinity
    idx = np.unique(([num for num, item in enumerate(itimes1) if np.any(np.isinf(item))]
                      + [num for num, item in enumerate(itimes2) if np.any(np.isinf(item))]))
    
    itimes1 = [item for num, item in enumerate(itimes1) if not num in idx]
    itimes2 = [item for num, item in enumerate(itimes2) if not num in idx]
            
    # return output
    return itimes1, itimes2



###############################################################################
def dist_to_exp(interarrival_times):
    
    ''' Calculates the distance between distributions: exponential distribution
    vs. distribution of interarrival times.
    
    Inputs:
        interarrival_times: interarrival times
        distance: distance measure, e.g. Wasserstein metric
    '''
    
    # get theoretical quantiles
    nobs = len(interarrival_times)
    theoretical = expon.ppf((np.arange(1.,nobs+1))/(nobs+1))
    
    # calculate distance between sample of theoretical vs. interarrival quantiles
    dist = wasserstein_distance(theoretical, interarrival_times)
        
    # return value
    return dist



###############################################################################
def plot_boxplots(itimes_list, model_names, plot_outliers=False,
                  save=False, saving_path=None, hour_list=None):
    ''' Plot boxplots.
    
    Inputs:
        itimes_list: list of interarrival times for different models
        model_names: list of model names
        plot_outliers: set to True for plotting outliers in boxplots
        save: set to True for saving graphics
        saving_path: specify path for saving graphics
        hour_list: list of hours if hourly evaluation is desired
    '''
    
    matplotlib.rcParams.update({'font.size': 12})
    
    if hour_list is None:
        # plot boxplots
        fig, ax = plt.subplots()
        
        plt.rcParams.update({'font.size': 12})
        
        ax.boxplot(itimes_list, labels=model_names, notch=True, bootstrap=10000,
                   showfliers=plot_outliers)
        
        #ax.set_ylim((0,1))
        ax.set_ylim((0,0.4))
        
        # save figure if desired
        if save:
            plt.savefig(saving_path + '/boxplot_disttoexp.png', dpi=300)
        
        # close figure
        plt.close()
        
    else:
        
        if not len(hour_list) % 3 == 0:
            raise ValueError('Number of hours cannot be divided by 4.')
        else:
            fig, ax = plt.subplots(nrows=int(len(hour_list)/3), ncols=3,
                                   sharey='all', figsize=(10,14))
            
            for hour, axis, key in zip(hour_list, ax.reshape(-1), itimes_list):
                
                data = itimes_list[key]
                
                axis.boxplot(data, labels=model_names, notch=True,
                             bootstrap=10000, showfliers=plot_outliers)
                
                axis.set_title('Hour ' + str(hour))
            
            plt.tight_layout()
            
            # save figure if desired
            if save:
                plt.savefig(saving_path + '/boxplot_disttoexp_hourly.png', dpi=300)
            
            # close figure
            plt.close()
            
            
            
###############################################################################
def do_distribution_test(itimes_path, hour_list=range(24), confidence_lvl=[95],
                         save=False, saving_path=None,
                         start_date=None, end_date=None, plot=False):
    ''' Run test for equality of distributions.
    
    Input:
        itimes_path: path of interarrival times
        hour_list: list of hours
        confidence_lvl: list of confidence levels
        save: set to True for saving results
        saving_path: specify path for saving results
        start_date: first model date
        end_date: last model date
        plot: set to True for plotting test results
    '''
    
    # check significance levels for ad test (only 15%, 10%, 5%, 2.5%, 1% levels are provided)
    inv_confidence_level = 100-np.array(confidence_lvl)
    
    for item in inv_confidence_level:
        if item in [15, 10, 5, 2.5, 1]:
            pass
        else:
            raise ValueError('Given significance level for AD test cannot be evaluated.')
    
    # load interarrival times
    itimes = tools.load_obj(itimes_path + '/interarrival_times.pkl', as_is=True)
    itimes = prepare_itimes(itimes, start_date, end_date)
    
    # get local timestamps
    times = pd.to_datetime(itimes["timestamp_UTC"], utc=True).tz_convert('Europe/Berlin')
    
    # preallocate dataframe
    df = pd.DataFrame(columns=[str(hour) for hour in hour_list],
                      index=["Models"]+["Tested"]+['p < ' + str(np.round((100-lvl)/100,4)) for lvl in confidence_lvl])
    
    # loop through hours
    for hour in hour_list:
        
        # get indices for respective hour
        idx = [num for num, date in enumerate(times) if date.hour==hour]
        
        # get list of interarrival times
        itimes_list = list(np.array(itimes["interarrival_times"])[idx])
        
        # get information on total models
        total = len(itimes_list)
        
        # write p-value of test in list 
        test_results = [anderson(x=item, dist='expon') for item in itimes_list]
        
        # count number of p-values below confidence levels
        significance_count = []
        for alpha in confidence_lvl:
            
            if len(test_results) == 0:
                significance_count.append(0)
            else:
                # get index for confidence level
                idx = test_results[0][2].tolist().index(100-alpha)
                
                significance_count.append(len([res[0] for res in test_results if res[0] > res[1][idx]]))
                
        # write information into dataframe
        df[str(hour)] = [int(total), int(len(test_results))] + list(significance_count)
        
    # plotting
    if plot:
        stacked_bars_test(dataframe=df, testtype="ad",
                          save=save, saving_path=saving_path)
        
    # save results
    if save:
        
        df.to_csv(saving_path + '/ad_test.csv', sep=';', index=True)
    
    # return output
    return df
        
        
        
###############################################################################
def do_correlation_test(itimes_path, hour_list=range(24), confidence_lvl=[95],
                        save=False, saving_path=None,
                        start_date=None, end_date=None, plot=False):
    ''' Testing correlations in interarrival times.
    
    Input:
        itimes_path: path of interarrival times
        hour_list: list of hours
        confidence_lvl: list of confidence levels
        save: set to True for saving results
        saving_path: specify path for saving results
        start_date: first model date
        end_date: last model date
        plot: set to True for plotting results
    '''
    
    # load interarrival times
    itimes = tools.load_obj(itimes_path + '/interarrival_times.pkl', as_is=True)
    itimes = prepare_itimes(itimes, start_date, end_date)
    
    # get local timestamps
    times = pd.to_datetime(itimes["timestamp_UTC"], utc=True).tz_convert('Europe/Berlin')
    
    # preallocate dataframe
    df = pd.DataFrame(columns=[str(hour) for hour in hour_list],
                      index=["Models", "Tested"]+['p < ' + str(np.round((100-lvl)/100,4)) for lvl in confidence_lvl])
    
    # loop through hours
    for hour in hour_list:
        
        # get indices for respective hour
        idx = [num for num, date in enumerate(times) if date.hour==hour]
        
        # get list of interarrival times
        itimes_list = list(np.array(itimes["interarrival_times"])[idx])
        
        # get information on total models
        total = len(itimes_list)
        
        # write p-value of test in list            
        lags = [5]
        
        test_results = []
        for item in itimes_list:
            try:
                test_results.append(acorr_ljungbox(x=item, lags=lags))
            except:
                print('Problem with conducting LB-test.')
        
        # count number of p-values below confidence levels
        significance_count = []
        for alpha in confidence_lvl:
            
            significance_count.append(len([res for res in test_results if res[1]<1-alpha/100]))

            
        # get information on tested models
        tested = len(test_results)
            
        # write information into dataframe
        df[str(hour)] = [int(total), int(tested)] + list(significance_count)
        
    # plotting
    if plot:
        stacked_bars_test(dataframe=df, testtype="lb",
                          save=save, saving_path=saving_path)
        
    # save results
    if save:
        
        df.to_csv(saving_path + '/lb_test.csv', sep=';', index=True)
        
    # return output
    return df
        
        

###############################################################################
def prepare_itimes(itimes, start_date, end_date):
    ''' Pick only desired dates from interarrival times.
    
    Input:
        itimes: dictionary of interarrival times
        start_date: start date
        end_date: end date
    '''
    
    if start_date is None and end_date is None:
        
        return itimes
    
    else:
    
        # get indices for desired dates
        if start_date is None:
            idx = [num for num, item in enumerate(itimes["timestamp_UTC"]) if item <= end_date]
        elif end_date is None:
            idx = [num for num, item in enumerate(itimes["timestamp_UTC"]) if item >= start_date]
        else:
            idx = [num for num, item in enumerate(itimes["timestamp_UTC"]) if item >= start_date and item <= end_date]
        
        # get corresponding dates and interarrival times
        dates = tuple(itimes["timestamp_UTC"][i] for i in idx)
        times = tuple(itimes["interarrival_times"][i] for i in idx)
        
        return {'timestamp_UTC': dates,
                'interarrival_times': times}
    
    
    
###############################################################################
def stacked_bars_test(dataframe, testtype, save=False, saving_path=None):
    ''' Barplot for Anderson-Darling or Ljung-Box test results.
    
    Inputs:
        dataframe: dataframe with test results
        testtype: 'ad' for Anderson-Darling, 'lb' for Ljung-Box
        save: set to True to save figures
        saving_path: path for saving figures
    '''
    
    # get row numbers of dataframe
    row_num = dataframe.shape[0]
    
    # get integers for hours
    hours = [int(item) for item in dataframe.columns]
    
    # predefine colorlists
    if testtype == 'ad':
        col_red = ['darkgreen', 'mediumseagreen', 'yellowgreen']
        col_blue = ['saddlebrown', 'peru', 'burlywood']
    elif testtype == 'lb':
        col_red = ['darkblue', 'steelblue', 'skyblue']
        col_blue = ['saddlebrown', 'peru', 'burlywood']
    else:
        raise ValueError('Unknown test.')
    
    # plot stacked bars
    matplotlib.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots()
    
    ax.bar(hours, dataframe.iloc[-1], color=col_red[0], label=dataframe.index[-1])
    
    if row_num > 3:
        todo = reversed(range(2,row_num-1))
        for row, col in zip(todo, col_red[1:]):
            ax.bar(hours, dataframe.iloc[row]-dataframe.iloc[row+1],
                   bottom=dataframe.iloc[row+1], color=col,
                   label=dataframe.index[row])
            
    # plot reference bars with model number
    ax.bar(hours, dataframe.iloc[1]-dataframe.iloc[2],
           bottom=dataframe.iloc[2], color='lightgray', label='Total models')

    
    # labels
    ax.set_xlabel('Hour')
    ax.set_ylabel('Number of models', color=col_red[0])
    ax.tick_params(axis='y', labelcolor=col_red[0])
    
    
    # second axis with percentage values
    ax2 = ax.twinx()
    ax2.set_ylabel('Percentage', color=col_blue[0])
    ax2.tick_params(axis='y', labelcolor=col_blue[0])
    
    ax2.plot(hours, (dataframe.iloc[2]/dataframe.iloc[1])*100, color=col_blue[0],
             linewidth=3, label=dataframe.index[2])
    ax2.set_ylim((0,100))
    
    if row_num > 3:
        todo = range(3,row_num)
        for row, col in zip(todo, col_blue[1:]):
            ax2.plot(hours, (dataframe.iloc[row]/dataframe.iloc[1])*100,
                     color=col, linewidth=3, label=dataframe.index[row])
            
            
    # custom legend
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    
    handles_all = []
    labels_all = []
    
    for label, handle in zip(labels1, handles1):
        labels_all.append(label)
        handle_match = [hand for lab,hand in zip(labels2,handles2)
                        if lab==label]
        if len(handle_match) > 0:
            handles_all.append((handle,handle_match[0]))
        else:
            handles_all.append((handle,))
    
    fig.legend(tuple(handles_all), tuple(labels_all), loc="lower left",
               bbox_to_anchor=(0,0.92,1,0), mode="expand", ncol=row_num-1)
    
    #plt.tight_layout()
    
    if save:
        plt.savefig(saving_path + '/' + testtype + '_barplot.png', dpi=300,
                    bbox_inches="tight")



#%% likelihood values
def difference_in_likelihoods(path_full, path_nested, save=False,
                              saving_path=None, model_names=None):
    ''' Calculates the differences in likelihood values of two nested models
    and plots the series of differences.
    
    Inputs:
		path_full: path of full model
		path_nested: path of nested model
		save: set to True to save results
		saving_path: path for saving results
		model_names: list of model names
    '''
    
    # get list of files for full model
    files_full = os.listdir(path_full)
    files_full = [file for file in files_full if file.endswith('.pkl')
                                              and not '_notConverged' in file]
    
    # get list of files for nested model
    files_nested = os.listdir(path_nested)
    files_nested = [file for file in files_nested if file.endswith('.pkl')
                                                  and not '_notConverged' in file]
    
    # get list of matched dates
    files, dates = match_files(files1=files_full, files2=files_nested)
    
    # loop through matched list and compute difference in likelihood values
    results = pd.DataFrame(index=dates, columns=['Difference'])
    for idx, item in zip(dates, files):
        
        # get results
        res_full = tools.load_obj(path_full + '/' + item[0], as_is=True)
        res_nested = tools.load_obj(path_nested + '/' + item[1], as_is=True)
        
        if res_full['success'] and res_nested['success']:
        
            # compute difference in likelihood values and write it to dataframe
            results.loc[idx, 'Difference'] = -res_full['fun']-res_nested['fun']
            
        else:
            
            results = results.drop(idx)
            
            
    # plot results
            
    # set date format
    date_fmt = mdates.DateFormatter('%Y-%m-%d')
    
    fig, ax = plt.subplots()
    
    plt.rcParams.update({'font.size': 12})
    
    ax.plot(mdates.date2num(results.index.values), results['Difference'].values)
    
    ax.xaxis.set_major_formatter(date_fmt)
    plt.xticks(rotation=45)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Difference in likelihood')
    
    ax.set_ylim((0,500))
    
    plt.tight_layout()
    
    if save:
        
        if model_names is None:
            plt.savefig(saving_path + '/likelihood_diff.png', dpi=300)
        else:
            plt.savefig(saving_path + '/likelihood_diff_' + model_names[0] + '_' + model_names[1] + '.png', dpi=300)
        
    # return output
    return results



      
#%% commands
if __name__ == '__main__':
    
    ######################### Parameter estimates #########################    

    ## full model   
    plot_estimates('../model_full/results/buy', alpha=95, save=True,
                   saving_path='../model_full/results/buy',
                   hour_list=[13])
    
    
      
    ######################### Likelihood ratio tests #########################
    
    #### full model vs. self-exciting model, buy orders (for sell orders change paths to results)
    do_lr_statistics(path_full='../model_full/results/buy',
                      path_nested='../model_selfexciting/results/buy',
                      confidence_lvl=[95, 97.5, 99],
                      save=True,
                      saving_path='../model_full/results/buy',
                      model_names=['Full', 'Selfexciting'],
                      hour_list=range(24),
                      plot=True)

    
        
    ######################## Interarrival times #########################
    
    ## compare distance to exponential
    
    # full model vs self-exciting model, buy orders (for sell orders change paths to results)
    # for hourly plots as in the appendix, set hour_list to range(24)
    
    compare_dist_to_exp(path_full='../model_full/results/buy/interarrival_times',
                        path_nested='../model_selfexciting/results/buy/interarrival_times',
                        model_names=['Full model', 'Self-exciting model'],
                        hour_list=None,
                        save=True,
                        saving_path='../model_full/results/buy/interarrival_times')
    
    # full model vs exogenous model, buy orders (for sell orders change paths to results)
    compare_dist_to_exp(path_full='../model_full/results/buy/interarrival_times',
                        path_nested='../model_exogenous/results/buy/interarrival_times',
                        model_names=['Full model', 'Exogenous model'],
                        hour_list=None,
                        save=True,
                        saving_path='../model_full/results/buy/interarrival_times')
    
    
    ## do distribution test for interarrival times
    
    # full model, buy orders (for sell orders change itimes_path and saving_path)
    do_distribution_test(itimes_path='../model_full/results/buy/interarrival_times',
                         hour_list=range(24),
                         confidence_lvl=[95, 97.5, 99],
                         save=True,
                         saving_path='../model_full/results/buy/interarrival_times',
                         plot=True)


      
    ## do autocorrelation test for interarrival times
    
    # full model, buy orders (for sell orders change itimes_path and saving_path)
    do_correlation_test(itimes_path='../model_full/results/buy/interarrival_times',
                        hour_list=range(24),
                        confidence_lvl=[90, 95, 99],
                        save=True,
                        saving_path='../model_full/results/buy/interarrival_times',
                        plot=True)
    
    
        
    ######################## Compare likelihood values #########################
    
    ### full model vs. exogenous model, buy orders (for sell orders change paths to results)
    difference_in_likelihoods(path_full='../model_full/results/buy',
                              path_nested='../model_exogenous/results/buy',
                              save=True,
                              saving_path='../model_full/results/buy',
                              model_names=['FullModel', 'ExogenousModel'])  
