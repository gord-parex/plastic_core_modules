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
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.colors import Normalize

import dash
from dash import dcc, html
import plotly.graph_objects as go
import dash.dependencies as dd
from dash import callback_context


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
    
    
    ########################################################################


def plot_log_scatter_data_with_kde(df, x_column, y_column, color_column=None, save_fig=False):
    if color_column is not None:
        df = df.sort_values(by=color_column)
        unique_colors = df[color_column].unique()
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, len(unique_colors)))
        
        # Separate data by categories for KDE plots
        x_data_list = [df[df[color_column] == unique_value][x_column] for unique_value in unique_colors]
        y_data_list = [df[df[color_column] == unique_value][y_column] for unique_value in unique_colors]
    else:
        unique_colors = None
        colors = ['gray']
        x_data_list = [df[x_column]]
        y_data_list = [df[y_column]]
    
    # Filter out nans in x_column and y_column
    x_data = df[x_column]
    y_data = df[y_column]
        
    figure_size = (20, 10)
    
    fig = plt.figure(layout='constrained', figsize=figure_size)
    
    # Main scatter plot
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    
    # Inset axes for KDE plots
    ax_kdex = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_kdey = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)  # Enable y-axis sharing

    # Plot scatter
    scatter = ax.scatter(x_data, y_data, c=df[color_column] if color_column is not None else 'gray', cmap='viridis', alpha=0.5)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)

    # Plot KDE plots, fill area under curves, and add mean lines
    for i, (x_data_single, y_data_single) in enumerate(zip(x_data_list, y_data_list)):
        sns.kdeplot(x=x_data_single, ax=ax_kdex, color=colors[i], alpha=0.7, bw_adjust=0.5, fill=True)
        sns.kdeplot(y=y_data_single, ax=ax_kdey, color=colors[i], alpha=0.7, bw_adjust=0.5, fill=True)
        
        # Calculate mean for x and y data
        x_mean = np.mean(x_data_single)
        y_mean = np.mean(y_data_single)
        
        # Plot vertical line for x mean
        ax_kdex.axvline(x=x_mean, color=colors[i], linestyle='--', linewidth=1, label=f'{unique_colors[i]} Mean' if unique_colors is not None else 'Mean')
        
        # Plot horizontal line for y mean
        ax_kdey.axhline(y=y_mean, color=colors[i], linestyle='--', linewidth=1)

    # Add legend to scatter plot if color_column is used
    if color_column is not None:
        handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
        ax.legend(handles, unique_colors, title=color_column, loc='upper right', fontsize='small')

    # Remove tick labels for the inset axes to make it visually clean
    ax_kdex.tick_params(axis='x', labelbottom=False)
    ax_kdex.tick_params(axis='y', labelleft=False)
    ax_kdey.tick_params(axis='x', labelbottom=False)
    ax_kdey.tick_params(axis='y', labelleft=False)
    
    # Adjust grid lines and set the face colors if desired
    ax.grid(True)
    
    # Optional: save the figure
    if save_fig:
        figure_name = f'{x_column}_vs_{y_column}_scatter_kde.png'
        plt.savefig(figure_name)

    plt.show()

    
def plot_log_scatter_data_with_histogram(df, x_column, y_column, color_column=None, save_fig=False, hist_stacked=True):
    if color_column is not None:
        df = df.sort_values(by=color_column)
        unique_colors = df[color_column].unique()
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, len(unique_colors)))
        
        # Separate data by categories for histograms
        x_data_list = [df[df[color_column] == unique_value][x_column] for unique_value in unique_colors]
        y_data_list = [df[df[color_column] == unique_value][y_column] for unique_value in unique_colors]
    else:
        unique_colors = None
        colors = ['gray']
        x_data_list = [df[x_column]]
        y_data_list = [df[y_column]]
    
    # Filter out nans in x_column and y_column
    x_data = df[x_column]
    y_data = df[y_column]
        
    figure_size = (20, 10)
    
    fig = plt.figure(layout='constrained', figsize=figure_size)
    
    # Main scatter plot
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    
    # Inset axes for histograms
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)  # Enable y-axis sharing

    # Plot scatter
    ax.scatter(x_data, y_data, c=df[color_column] if color_column is not None else 'gray', cmap='viridis', alpha=0.5)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)

    # Plot histograms
    if hist_stacked:
        # Plot stacked histograms
        ax_histx.hist(x_data_list, bins=100, color=colors, alpha=0.7, stacked=True)
        ax_histy.hist(y_data_list, bins=100, orientation='horizontal', color=colors, alpha=0.7, stacked=True)
    else:
        # Plot overlaid histograms with transparency
        for i, (x_data_single, y_data_single) in enumerate(zip(x_data_list, y_data_list)):
            ax_histx.hist(x_data_single, bins=100, density='normal', color=colors[i], alpha=0.7, label=f'{unique_colors[i]}')
            ax_histy.hist(y_data_single, bins=100, density='normal', orientation='horizontal', color=colors[i], alpha=0.7)

        # Add legend to histograms to show color meaning if not stacked
        ax_histx.legend(loc='upper right', fontsize='small')

    # Remove tick labels for the inset axes to make it visually clean
    ax_histx.tick_params(axis='x', labelbottom=False)
    ax_histx.tick_params(axis='y', labelleft=False)
    ax_histy.tick_params(axis='x', labelbottom=False)
    ax_histy.tick_params(axis='y', labelleft=False)
    
    # Adjust grid lines and set the face colors if desired
    ax.grid(True)
    
    # Optional: save the figure
    if save_fig:
        figure_name = f'{x_column}_vs_{y_column}_scatter_histogram.png'
        plt.savefig(figure_name)

    plt.show()
    
  
    
    
    ################################################

 
def create_multiple_plots(df, step_list=[3], reservoir_unit_list=None):
    for num_steps in step_list:
        pl.calculate_rate_of_change_curve(mega_df,'RESD_LOG10', num_depth_steps=num_steps)
        pl.calculate_rate_of_change_curve(mega_df,'RHOB', num_depth_steps=num_steps)
        pl.calculate_rate_of_change_curve(mega_df,'GR', num_depth_steps=num_steps)
        
        for unit in reservoir_unit_list:
            plot_derivative_log_data_with_histogram(mega_df, steps=num_steps, screening_formation=unit)
            
            

def histogram_plot_select_2D(dataframe, x_col, y_col, num_bins=300, opacity_value=0.1, port=8050):
    # Make a copy of the dataframe to prevent modifying the original one
    df = dataframe.copy()

    # Create df['Flag'] or set it to 0
    df['Flag'] = 0

    # Create the Dash app
    app = dash.Dash(__name__)

    # Layout
    app.layout = html.Div([
        dcc.Graph(id='scatter-plot', config={'modeBarButtonsToAdd': ['lasso2d', 'select2d']}),
        dcc.Store(id='selected-points'),  # Hidden store to keep selected points data
        html.Div([
            dcc.Input(id='flag-value', type='number', placeholder='Enter flag value', value=1, style={'margin-right': '10px'}),
            html.Button('Apply Flag', id='apply-flag', n_clicks=0)
        ], style={'margin-bottom': '10px'}),
        dcc.Input(id='num-bins', type='number', placeholder='Enter number of bins', value=num_bins),
        
        html.Button('Complete Flagging', id='complete-flagging', n_clicks=0),
        html.Div(id='confirmation-message', style={'margin-top': '10px', 'font-weight': 'bold'}),
        html.Div(id='selection-info', style={'margin-top': '10px'})
    ])

    # Callback to update the selected points in the store
    @app.callback(
        dd.Output('selected-points', 'data'),
        [dd.Input('scatter-plot', 'selectedData')]
    )
    def store_selected_points(selectedData):
        if selectedData:
            return selectedData['points']
        return []

    # Callback to update the scatter plot
    @app.callback(
        dd.Output('scatter-plot', 'figure'),
        [dd.Input('apply-flag', 'n_clicks'),
         dd.Input('num-bins', 'value')],
        [dd.State('selected-points', 'data'), dd.State('flag-value', 'value')]
    )
    def update_scatter_plot(n_clicks, num_bins, selected_points, flag_value):
        # Update the DataFrame based on the selected data
        if n_clicks > 0 and selected_points:
            for point in selected_points:
                x_val, y_val = point['x'], point['y']
                df.loc[(df[x_col] == x_val) & (df[y_col] == y_val), 'Flag'] = flag_value

        # Create the 2D histogram with Plotly (non-interactive layer)
        histogram = go.Histogram2d(
            x=df[x_col],
            y=df[y_col],
            colorscale='Reds',
            nbinsx=num_bins,
            nbinsy=num_bins,
            opacity=0.5,
            showscale=False  # Make the histogram non-interactive
        )

        # Create the scatter plot (interactive layer)
        scatter = go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='markers',
            marker=dict(
                size=5,
                color=df['Flag'],  # Use the updated flag column for coloring
                colorscale='Viridis',
                showscale=True,
                opacity=opacity_value
            )
        )

        # Combine the 2D histogram and scatter plot
        fig = go.Figure(data=[histogram, scatter])
        fig.update_layout(
            title='Interactive Scatter Plot with 2D Histogram Background',
            xaxis_title=x_col,
            yaxis_title=y_col,
            autosize=False,
            width=800,
            height=600,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        return fig

    @app.callback(
        [dd.Output('confirmation-message', 'children'), dd.Output('selection-info', 'children')],
        [dd.Input('apply-flag', 'n_clicks'), dd.Input('complete-flagging', 'n_clicks')],
        [dd.State('selected-points', 'data'), dd.State('flag-value', 'value')]
    )
    def confirmation_message(apply_n_clicks, complete_n_clicks, selected_points, flag_value):
        selection_count = len(selected_points) if selected_points else 0
        selection_info = f'Selected points: {selection_count}'
        
        if apply_n_clicks > 0:
            return f'Flag applied with value: {flag_value}. Updated {selection_count} points.', selection_info
        elif complete_n_clicks > 0:
            return f'Flagging complete. Returning modified DataFrame.', selection_info
        return '', selection_info

    # Run the app
    app.run_server(debug=True, port=port)

    # Return the modified DataFrame
    return df

  
   


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
    
   
