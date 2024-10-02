# -*- coding: utf-8 -*-
"""
Created on Wed May  1 17:01:02 2024

@author: Gfoo
"""

"""This code was created to plot histograms, and scatter plots that have
histograms on their margins.

An example at the bottom shows how the "derivative" curves of gamma ray,
resistivity and density cross plot against one another.

"""

import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.colors import Normalize

from . import plastic as pl

def create_custom_viridis():
    viridis = cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    white = np.array([1, 1, 1, 1])
    newcolors[:1, :] = white  # Set the first color entry to white
    newcmap = ListedColormap(newcolors)
    return newcmap

# def separate_dataframe_oil_show(df):
#     df1 = df[df['OIL_SHOW'] == 0]
#     df2 = df[df['OIL_SHOW'] == 0.25]
#     df3 = df[df['OIL_SHOW'] == 0.5]
#     df4 = df[df['OIL_SHOW'] == 0.75]
#     df5 = df[df['OIL_SHOW'] == 1]
    
#     return (df1, df2, df3, df4, df5)

def plot_derivative_log_data(df, steps=3, screening_formation=None, color_order=False, color_column=None):
    
    
    if screening_formation != None:
        df = df[df['Strat_unit_name'] == screening_formation]
    if color_order == True:
        try:
            assert color_column != None
        except AssertionError as a:
            print(f'When declaring color order, a color column must be specified')
            return a
        df.sort_values(by=color_column)
    
    
    rhob_series = df['RHOB_ra']#']
    gr_series = df['GR_rate']
    resdlog10_series =  df['RESD_LOG10']
    oil_show_series = df['OIL_SHOW']
    color_column_series = df[color_column]
    
    plt.figure(figsize=(40,20))
    plt.scatter(rhob_series, resdlog10_series, c=color_column_series, cmap='viridis', s=0.3, facecolor='grey', alpha=0.7)
    plt.xlim(-0.5,0.5)
    plt.ylim(-1.2,1.2)
    figure_name = f'RHOB_ResDLog10_{steps}_steps_{screening_formation}_fmn.png'
    
    plt.savefig(figure_name)
    
    plt.figure(figsize=(40,20))
    plt.scatter(gr_series, resdlog10_series, c=color_column_series, cmap='viridis', s=0.3, facecolor='grey', alpha=0.7)
    plt.xlim(-100,100)
    plt.ylim(-1.2,1.2)
    figure_name = f'GR_ResDLog10_{steps}_steps_{screening_formation}_fmn.png'
    
    plt.savefig(figure_name)
    
def scatter_hist(x, y, ax, ax_histx, ax_histy, color_data=None):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=True)
    ax_histy.tick_params(axis="y", labelleft=True)
    
    # Create a new colormap from viridis starting with white
    
    custom_viridis = create_custom_viridis()
    
    norm = Normalize(vmin=np.min(color_data), vmax=np.max(color_data))
    colormap = custom_viridis
    
    num_bins = 500    
    
    if color_data is not None:
        sc = ax.scatter(x, y, s=0.5, c=color_data, cmap=colormap, label='Color', facecolor='grey')
        plt.xlabel('X_Data')
        plt.ylabel('Y_Data')
        plt.colorbar(sc, ax=ax, label='Oil Show')
        
        
        unique_colors = np.unique(color_data)
        
        # Prepare data for histograms
        color_bins_x = {}
        color_bins_y = {}
        
        for color in unique_colors:
            color_bins_x[color] = x[color_data == color]
            color_bins_y[color] = y[color_data == color]
        
        #Do not include zero values, index from second item (index 1)
        for color in unique_colors[1:]:
            ax_histx.hist(color_bins_x[color], bins=num_bins, histtype='bar', stacked=True, color=colormap(norm(color)))
            ax_histy.hist(color_bins_y[color], bins=num_bins, histtype='bar', stacked=True, color=colormap(norm(color)), orientation='horizontal')
    else:
        sc = ax.scatter(x, y)#, facecolor='grey')
        ax_histx.hist(x, bins=num_bins, histtype='bar')
        ax_histy.hist(x, bins=num_bins, histtype='bar', orientation='horizontal')
        
 
    
 
    
def plot_derivative_log_data_with_histogram(df, steps=3, screening_formation=None, color_order=False, color_column=None):
    if screening_formation != None:
        df = df[df['Strat_unit_name'] == screening_formation]
    if color_order == True:
        try:
            assert color_column != None
        except AssertionError as a:
            print(f'When declaring color order, a color column must be specified')
            return a
        df = df.sort_values(by=color_column)
    
    #Filter out nans in color_column column
    df = df.dropna(subset=[color_column])
    
    rhob_series = df['RHOB_rate']
    gr_series = df['GR_rate']
    resdlog10_series =  df['RESD_LOG10_rate']
    oil_show_series = df['OIL_SHOW']
    color_column_series = df[color_column]
    
    
    
    figure_size=(40,20)
    
    fig = plt.figure(layout='constrained', figsize=figure_size)
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    ax.set_facecolor('grey')
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    scatter_hist(rhob_series, resdlog10_series, ax, ax_histx, ax_histy, color_column_series)
    plt.xlim(-0.5,0.5)
    plt.ylim(-1.2,1.2)
    figure_name = f'RHOB_ResDLog10rate_{steps}_steps_{screening_formation}_fmn_histogram.png'
    plt.savefig(figure_name)
    
    
    fig = plt.figure(layout='constrained', figsize=figure_size)
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    ax.set_facecolor('grey')
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    scatter_hist(gr_series, resdlog10_series, ax, ax_histx, ax_histy, color_column_series)
    plt.xlim(-100,100)
    plt.ylim(-1.2,1.2)
    figure_name = f'GR_ResDLog10rate_{steps}_steps_{screening_formation}_fmn_histogram.png'
    plt.savefig(figure_name)
    
    
    
def plot_log_scatter_data_with_histogram(df, x_column, y_column, color_column=None):
    
    if color_column is not None:
        try:
            assert color_column != None
        except AssertionError as a:
            print(f'When declaring color order, a color column must be specified')
            return a
        df = df.sort_values(by=color_column)
        color_data = df[color_column]
    else:
        color_data = None
    
    #Filter out nans in color_column column
    x_data = df[x_column]
    y_data = df[y_column]
        
    figure_size=(20,10)
    
    fig = plt.figure(layout='constrained', figsize=figure_size)
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    # ax.set_facecolor('grey')
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    scatter_hist(x_data, y_data, ax, ax_histx, ax_histy, color_data)
    # plt.xlim(-0.5,0.5)
    # plt.ylim(-1.2,1.2)
    figure_name = f'{x_column}_vs_{y_column}_scatter_histogram.png'
    plt.savefig(figure_name)
    
    

 
def create_multiple_plots(df, step_list=[3], reservoir_unit_list=None):
    for num_steps in step_list:
        pl.calculate_rate_of_change_curve(mega_df,'RESD_LOG10', num_depth_steps=num_steps)
        pl.calculate_rate_of_change_curve(mega_df,'RHOB', num_depth_steps=num_steps)
        pl.calculate_rate_of_change_curve(mega_df,'GR', num_depth_steps=num_steps)
        
        for unit in reservoir_unit_list:
            plot_derivative_log_data_with_histogram(mega_df, steps=num_steps, screening_formation=unit)
 
    
   


if __name__ == "__main__":
    
    mega_df_fn = r"C:\Programming\AI_Production_Log_Project_2023\Scripting Session Folders\22APR2024 Bypassed Pay Session\Logs\Mega Dataframe\With tops and coordinates and log10 curves\mega_df_with_tops_and_coordinates_25APR2024.pkl"
    
    
    #Load mega_df, the main database for log data.
    with open(mega_df_fn, 'rb') as f:
        mega_df = pkl.load(f)

        
    #Create derivative curves.
    pl.calculate_rate_of_change_curve(mega_df,'RESD_LOG10')
    pl.calculate_rate_of_change_curve(mega_df,'RHOB')
    pl.calculate_rate_of_change_curve(mega_df,'GR')
    pl.calculate_rate_of_change_curve(mega_df,'C3_GAS_LOG10')
    
    #Create series for each calculated curve.
    rhob_series = mega_df['RHOB_rate']
    gr_series = mega_df['GR_rate']
    resdlog10_series =  mega_df['RESD_LOG10_rate']
    C3_gas_show_series = mega_df['C3_GAS_LOG10_rate']
    
    # reservoir_unit_list = ['Carbonera 3', 'Carbonera 5', 'Carbonera 7', 
    #                        'Mirador', 'Barco', 'Guadalupe', 'Gacheta', 'Une', None]
    
    reservoir_unit_list = ['Carbonera 3', 'Carbonera 5', 'Carbonera 7']
    
    #This is a list telling the script across how many rows it should calculate
    #the "slope" or "derivative" that becomes the new calculated curve.
    step_list = [2, 4, 8]
    
    
    # for num_steps in step_list:
    #     pl.calculate_rate_of_change_curve(mega_df,'RESD_LOG10', num_depth_steps=num_steps)
    #     pl.calculate_rate_of_change_curve(mega_df,'RHOB', num_depth_steps=num_steps)
    #     pl.calculate_rate_of_change_curve(mega_df,'GR', num_depth_steps=num_steps)
        
    #     for unit in reservoir_unit_list:
    #         plot_derivative_log_data_with_histogram(mega_df, steps=num_steps, screening_formation=unit)
    
    #Filter out rows that only have specified strat_unit_name
    super_df = mega_df[mega_df['Strat_unit_name'].isin(reservoir_unit_list)]
    
    #Filter on low ResD
    super_df = super_df[super_df['RESD'] < 40]
    
    plot_derivative_log_data_with_histogram(super_df, steps=3, color_order=True, color_column='OIL_SHOW')
    
   
