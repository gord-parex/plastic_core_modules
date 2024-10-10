# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 08:43:48 2024

@author: gfoo
"""

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import plotly.io as pio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.cluster import KMeans
pio.renderers.default = 'browser'
import pdb

from . import plastic as pl
from . import meta_plastic
from . import curve_plotting


class MLmodelling:
    def __init__(self):
        self.lithology_curves =  ['GR', 'RHOB', 'NPHI', 'PEF', 'DT', 'DTS', 'SP']
        self.fluid_lithology_curves = [
                                'GR',
                                'RESS_LOG10',
                                'RESM_LOG10',
                                'RESD_LOG10',
                                'SP',
                                'C1_GAS_LOG10',
                                'C2_GAS_LOG10',
                                'C3_GAS_LOG10',
                                'IC4_GAS_LOG10',
                                'IC5_GAS_LOG10',
                                'NC4_GAS_LOG10',
                                'NC5_GAS_LOG10',
                                'TOTALGAS_LOG10',
                                'OIL_SHOW_BINARY',
                                'RHOB',
                                'NPHI',
                                'PEF',
                                'CALI',
                                'DRHO',
                                'DT',
                                'DTS',
                                'Y_COORD',
                                'X_COORD',
                                'DELTA_PHID_PHIN',
                                'Haworth_fluid_classification',
                                'CHARACTER_RATIO', 
                                'WETNESS_RATIO',
                                'BALANCE_RATIO']

        self.continuous_numeric_reservoir_curves = [
                                'GR',
                                'RESS_LOG10',
                                'RESM_LOG10',
                                'RESD_LOG10',
                                'SP',
                                'C1_GAS_LOG10',
                                'C2_GAS_LOG10',
                                'C3_GAS_LOG10',
                                'IC4_GAS_LOG10',
                                'IC5_GAS_LOG10',
                                'NC4_GAS_LOG10',
                                'NC5_GAS_LOG10',
                                'TOTALGAS_LOG10',
                                'RHOB',
                                'NPHI',
                                'PEF',                                
                                'DRHO',
                                'DT',
                                'DTS',                                
                                'DELTA_PHID_PHIN',                                
                                'CHARACTER_RATIO', 
                                'WETNESS_RATIO',
                                'BALANCE_RATIO']
        
        self.continuous_numeric_reservoir_curves_no_res = [
                                'GR',
                                'SP',
                                'C1_GAS_LOG10',
                                'C2_GAS_LOG10',
                                'C3_GAS_LOG10',
                                'IC4_GAS_LOG10',
                                'IC5_GAS_LOG10',
                                'NC4_GAS_LOG10',
                                'NC5_GAS_LOG10',
                                'TOTALGAS_LOG10',
                                'RHOB',
                                'NPHI',
                                'PEF',                                
                                'DRHO',
                                'DT',
                                'DTS',                                
                                'DELTA_PHID_PHIN',                                
                                'CHARACTER_RATIO', 
                                'WETNESS_RATIO',
                                'BALANCE_RATIO']

    def pca_visuals(self, df, pca_object, vis_components):
        # Visualizing the explained variance ratio
        plt.figure(figsize=(8, 6))
        plt.plot(np.cumsum(pca_object.explained_variance_ratio_), marker='o')
        plt.title('Explained Variance by Number of Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid(True)
        plt.show()
    
        feature_names = df.columns
    
        # Radar plot of PCA components
        pca_components = pca_object.components_[:vis_components]
    
        angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
        angles += angles[:1]
    
        # Create subplots: one radar chart per PCA component
        num_components = len(pca_components)
        fig, axes = plt.subplots(1, num_components, figsize=(6 * num_components, 6),
                                 subplot_kw=dict(polar=True))
        
        # If there's only one component, axes won't be a list, so we wrap it in one
        if num_components == 1:
            axes = [axes]
    
        for i, (component, ax) in enumerate(zip(pca_components, axes)):
            values = component.tolist()
            values += values[:1]  # Ensure the plot is closed (returns to the start)
    
            # Plot radar chart for each component
            ax.plot(angles, values, label=f'PCA {i+1}')
            ax.fill(angles, values, alpha=0.1)
    
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(feature_names, size=10)
            ax.set_rlabel_position(30)
            ax.set_title(f'PCA {i+1} Radar Plot')
    
        plt.tight_layout()
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.show()


        
    # def pca_visuals(self, df, pca_object, vis_components):
    #     # Visualizing the explained variance ratio
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(np.cumsum(pca_object.explained_variance_ratio_), marker='o')
    #     plt.title('Explained Variance by Number of Components')
    #     plt.xlabel('Number of Components')
    #     plt.ylabel('Cumulative Explained Variance')
    #     plt.grid(True)
    #     plt.show()
        
    #     feature_names = df.columns
        
    #     # Radar plot of PCA components
    #     pca_components = pca_object.components_
        
    #     angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
    #     angles += angles[:1]
        
    #     fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    #     for i, component in enumerate(pca_components):
    #         values = component.tolist()
    #         values += values[:1]
    #         ax.plot(angles, values, label=f'PCA {i+1}')
    #         ax.fill(angles, values, alpha=0.1)
        
    #     ax.set_xticks(angles[:-1])
    #     ax.set_xticklabels(feature_names, size=10)
    #     ax.set_rlabel_position(30)
    #     ax.set_title('PCA Component Radar Plot')
    #     plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    #     plt.show()
    
    def pca_analysis(self, df, scaler=None, visuals=True, df_components=4, vis_components=4):
        #Drop non-numeric columns
        df_numeric = df.select_dtypes('number')
        
        #Check for null values, which are not permitted in PCA
        if df.isnull().values.any():
            print('Dataframe contains nans. Please check input data')
            return None
        
        
        if scaler == None:
            scaler = StandardScaler()
        else:
            scaler = scaler

        scaled_df = scaler.fit_transform(df_numeric)

        X = scaled_df
        
        pca = PCA()

        pca.fit(X)

        X_pca = pca.transform(X)
        
        if visuals == True:
            self.pca_visuals(df, pca, vis_components)
        
        X_pca_df = pd.DataFrame(X_pca)
        
        PCA_dict = {}
        for i in range(df_components):
            num = i + 1
            PCA_dict_key = f'PC_{num}'
            
            PCA_dict[PCA_dict_key] = X_pca_df[i]
        
        df_final = df.copy()
        
        df_final = df_final.reset_index()
        
        for PC_name, PC in PCA_dict.items():
            df_final[PC_name] = PC
            
            #print(df_final)
            
        df_final = df_final.set_index(['Well_name', 'DEPT'])
        
        return (X_pca, df_final)
    
    
   
    
    
    def record_elbow_plot(self, df):
        # Original DataFrame
               
        # List to store results: Num_Parameters, Num_Valid_Rows, and the columns included
        results = []
        
        # Initial list of all columns
        columns = df.columns.tolist()

        column_to_drop = None
        
        # Loop through, dropping one column with the most NaNs at a time
        for i in range(len(columns)):
            # Drop rows with NaNs in the current set of columns
            valid_rows = df[columns].dropna()
            num_valid_rows = valid_rows.shape[0]
            
            # Store number of parameters, number of valid rows, and the columns in use
            results.append((len(columns), num_valid_rows, columns.copy(), column_to_drop))
            
            # Drop the column with the most NaNs for the next iteration
            nan_counts = df[columns].isna().sum()
            column_to_drop = nan_counts.idxmax()
            columns.remove(column_to_drop)
        
        # Convert results to DataFrame for easy analysis
        result_df = pd.DataFrame(results, columns=["Num_Parameters", "Num_Valid_Rows", "Included_Columns", "Column Dropped"])
        
        # Print the DataFrame to view all the steps
        print(result_df)
        
        # Plot the trade-off between number of parameters and number of valid rows
        plt.plot(result_df['Num_Parameters'], result_df['Num_Valid_Rows'])
        plt.xlabel('Number of Parameters')
        plt.ylabel('Number of Valid Rows')
        plt.title('Trade-off: Number of Parameters vs Valid Rows')
        plt.show()
        
        return result_df

    def correlation_matrix(self, df):
        correlation = df.corr()

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(correlation, dtype=bool))
        
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))
        
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    def kmeans(self, df, columns=None, n_clusters=8):
        
        clusterer = KMeans(n_clusters=n_clusters)
        
        if columns != None:
            df_cluster = df[columns]
        
        cluster_labels = clusterer.fit_predict(df_cluster)
        
       
        df['Cluster_label'] = cluster_labels
        
        return df
        
        
      
    def create_3D_PCA_scatter(self, data, axis_labels=None, **kwargs):
        """Set up a 3D plotting function to accept an arbitrary number of data
         sets"""
    
        print(f'kwargs {kwargs}')
        
        def _get_scatter(df, **kwargs):
            x_data = df['PC_1']
            y_data = df['PC_2']
            z_data = df['PC_3']
                        
            # x_data, y_data, z_data, *remaining = data
    
            # # If the remaining elements are a dictionary (or similar), treat it as **kwargs
            # kwargs = remaining[0] if remaining and isinstance(remaining[0], dict) else {}
    
            #Retrieve any passed values from kwargs
            mode_value = kwargs.get('mode', 'markers')
            size_value = kwargs.get('size', 2)
            color_value = kwargs.get('color', None)
            color_scale = kwargs.get('color_scale', 'Rainbow')
            print(color_value)
            opacity_value = kwargs.get('opacity', 1)
            name_value = kwargs.get('name', '')
            
            if color_value is not None:
                color_data = df['Cluster_label']
                print(f'Color data {color_data}')
            else:
                color_data = None
            
            trace = go.Scatter3d(x=x_data,
                                 y=y_data, 
                                 z=z_data,
                                 name=name_value,
                                 mode=mode_value,
                                 marker=dict(
                                     size=size_value,
                                     color=color_data,
                                     colorscale=color_scale,
                                     opacity=opacity_value
                                 )
                                )
            
            return trace
        
         
        traces = []
        
        
        

        
        if ((len(data) == 1) | (type(data) == pd.DataFrame)):
            trace = _get_scatter(data, **kwargs)
            traces.append(trace)
        elif len(data) > 1: 
            for data_set in data:
                trace = _get_scatter(data_set, **kwargs)
                traces.append(trace)
         
        if axis_labels == None:
             x_axis_label = 'X Axis'
             y_axis_label = 'Y Axis'
             z_axis_label = 'Z Axis'
             
        else:
            x_axis_label, y_axis_label, z_axis_label = axis_labels
    
    
        # Set up the layout
        layout = go.Layout(scene=dict(
            xaxis_title = x_axis_label,
            yaxis_title = y_axis_label,
            zaxis_title = z_axis_label
        ),
                           width=1200,
                           height=1200
                          )
        
        # Combine data and layout into a figure
        fig = go.Figure(data=traces, layout=layout)
        
    
        # Show the plot
        fig.show()




    def create_3D_scatter(self, data, x_col=None, y_col=None, z_col=None, 
                          axis_labels=None, secondary_mkr_size=10,  **kwargs):
        """Set up a 3D plotting function to accept an arbitrary number of data
         sets"""
    
        print(f'kwargs {kwargs}')
        
              
        # if x_col == None:
        #     x_col = 'PC_1'
        # if y_col == None:
        #     y_col = 'PC_2'
        # if z_col == None:
        #     z_col = 'PC_3'
        
        x_log = kwargs.get('x_log', False)
        y_log = kwargs.get('y_log', False)
        z_log = kwargs.get('z_log', False)
        
        #Check if the x_col, y_col and z_col parameters have been assigned,
        #first check for Principal component labels in df.columns,
        #otherwise, index first 3 columns to plot
        
        def _assign_column_labels(df, x_col, y_col, z_col):
            
            if x_col == None:
                if 'PC_1' in df.columns:
                    x_col = 'PC_1'
                else:
                    x_col = df.columns[0]
                    print(f'Using {x_col} default for x axis column')
                    
            if y_col == None:
                if 'PC_2' in df.columns:
                    y_col = 'PC_2'
                else:
                    y_col = df.columns[1]
                    print(f'Using {y_col} default for y axis column')
                    
            if z_col == None:
                if 'PC_3' in df.columns:
                    z_col = 'PC_3'
                else:
                    z_col = df.columns[2]
                    print(f'Using {z_col} default for z axis column')
                    
            return (x_col, y_col, z_col)
                
        
        def _get_scatter(df, **kwargs):
                       
            x_data = df[x_col]
            y_data = df[y_col]
            z_data = df[z_col]
                        
            # x_data, y_data, z_data, *remaining = data
    
            # # If the remaining elements are a dictionary (or similar), treat it as **kwargs
            # kwargs = remaining[0] if remaining and isinstance(remaining[0], dict) else {}
    
            #Retrieve any passed values from kwargs
            mode_value = kwargs.get('mode', 'markers')
            size_value = kwargs.get('size', 2)
            color_value = kwargs.get('color', None)
            color_scale = kwargs.get('color_scale', 'Rainbow')
            print(color_value)
            opacity_value = kwargs.get('opacity', 1)
            name_value = kwargs.get('name', '')
            
            #TODO Fix this to accept a column label, or a string that
            #corresponds to a color
            
            # if color_value is not None:
            #     color_data = df['Cluster_label']
            #     print(f'Color data {color_data}')
            # else:
            #     color_data = None
            
            trace = go.Scatter3d(x=x_data,
                                 y=y_data, 
                                 z=z_data,
                                 name=name_value,
                                 mode=mode_value,
                                 marker=dict(
                                     size=size_value,
                                     color=color_value,
                                     colorscale=color_scale,
                                     opacity=opacity_value
                                 )
                                )
            
            return trace
        
         
        traces = []
        
         
        if ((len(data) == 1) | (type(data) == pd.DataFrame)):
            #Index single element if data is a list
            if type(data) == list:
                data = data[0]
            
            #Assign column labels on dataframe
            x_col, y_col, z_col = _assign_column_labels(data, x_col, y_col, z_col)
            
            #Get a scatter plot
            trace = _get_scatter(data, **kwargs)
            
            #Append to trace list
            traces.append(trace)
        
        #If data is a list with more than one element
        elif len(data) > 1:
            #Assign column labels on first dataframe
            x_col, y_col, z_col = _assign_column_labels(data[0], x_col, y_col, z_col)
            
            #Assign column labels to a list for assertion check later on
            required_columns = [x_col, y_col, z_col]
            
            #Loop over dataframes
            for idx, data_set in enumerate(data):
                required_columns = [x_col, y_col, z_col]
                #Assertion check to see that that x_col, y_col and z_col are all in the dataframe
                assert set(required_columns).issubset(data_set.columns), "Required columns are missing"
                
                #Once the first element is plotted (should be background data),
                #start plotting additional data sets with different sized
                #markers and different coloured markers
                if idx > 0:
                    kwargs['size'] = secondary_mkr_size
                    kwargs['color'] = np.random.rand()
                
                #Get a scatter plot
                trace = _get_scatter(data_set, **kwargs)
                
                #Append to trace list
                traces.append(trace)
        
        #Assign column labels
        if axis_labels == None:
             x_axis_label = x_col
             y_axis_label = y_col
             z_axis_label = z_col
             
        else:
            x_axis_label, y_axis_label, z_axis_label = axis_labels
    
    
        # Set up the layout
        layout = go.Layout(scene=dict(
            xaxis_title = x_axis_label,
            yaxis_title = y_axis_label,
            zaxis_title = z_axis_label
        ),
                           width=1200,
                           height=1200
                          )
        
        # Combine data and layout into a figure
        fig = go.Figure(data=traces, layout=layout)
        
        #Change axis types to logarithmic, if specified.
        if x_log:
            fig.update_layout(scene=dict(xaxis=dict(dtick=1, type='log')))
        
        if y_log:
            fig.update_layout(scene=dict(yaxis=dict(dtick=1, type='log')))
            
        if z_log:
            fig.update_layout(scene=dict(zaxis=dict(dtick=1, type='log')))
    
        # Show the plot
        fig.show()


        
    def get_color_column_df_plotly(self, df, cluster_label_col):
        cluster_label_series = df[cluster_label_col]
        
        unique_categories = list(cluster_label_series.unique())
        
        n = len(unique_categories)
        
        colormap = cm.get_cmap('hsv', n)
        category_color_map = {category: colormap(i) for i, category in enumerate(unique_categories)}
        
        # df['Color'] = df[cluster_label_col].astype(str)
        
        df['Color'] = df['Cluster_label'].map(category_color_map)

        return df                 
        
     #%%   
        
if __name__ == "__main__":
    
    consolidated_data_df_fn = r'C:\Programming\AI_Production_Log_Project_2023\Scripting Session Folders\22APR2024 Bypassed Pay Session\Consolidated_Well_Info 08AUG2024.xlsx'



    mega_df = pd.read_pickle(r"C:\Programming\AI_Production_Log_Project_2023\Scripting Session Folders\10JUN2024 Scripting Session\mega_df_tops_xcoord_logCurves_gasRatio_haworthClassfn_oilShowBinary_nphiCleaned_curveLimits_deltaPhinPhid_perfs_09SEP2024.pkl")

#%%

    las_maracas_wells = meta_plastic.LogDataFilter.well_name_filter(mega_df, ['LMAM'])

    las_maracas_gacheta = meta_plastic.LogDataFilter.formation_filter(las_maracas_wells, ['Gacheta'])


    MLmodel = MLmodelling()
    
    df_data = las_maracas_gacheta[MLmodel.lithology_curves]
    
    records_info = MLmodel.record_elbow_plot(df_data)
    
    df_data = df_data.drop(['SP', 'DT', 'DTS'], axis=1)
    
    df_data = df_data.dropna(how='any')
#%%

    pca_raw_data, pca_df = MLmodel.pca_analysis(df_data)
    
#%%
    
    #MLmodel.create_3D_scatter(pca_df)
    

#%%

    result_df = MLmodel.kmeans(pca_df, columns=['PC_1', 'PC_2', 'PC_3', 'PC_4'])


#%%
    
    result_df_color = MLmodel.get_color_column_df_plotly(result_df, 'Cluster_label')
    
#%%
    
    
    MLmodel.create_3D_scatter(result_df_color, color_value=result_df_color['Color'])    