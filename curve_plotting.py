# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:48:01 2024

@author: Gfoo
"""

import os

#Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle
from matplotlib import cm as cmap
import pdb


from . import plastic as pl
from . import meta_plastic
from . import merge_multiple_las_gas_electric_logs as merge

#Ensure these point to the local directory
logs_dir = os.getcwd() + r'\Logs' #Logs directory is subfolder of current working directory. Concatenate these strings to create path


#To assign each curve to a bin, with the particular parameters of the curve
curve_to_category_dict = {'curve_category_1a': {'name': 'Gamma Ray', 'bin': 'Bin01', 'xmin': 0, 'xmax': 200, 'unit': 'gapi', 'color':'green', 'fill': True, 'fill_direction': 'right_to_left', 'fill_alpha': 1, 'curve_mnems':['GR', 'GR1']},
            'curve_category_01b': {'name': 'Vshale', 'bin': 'Bin01', 'xmin': 0, 'xmax': 1, 'unit': 'frac', 'color': 'black', 'fill': False, 'fill_direction': 'right_to_left', 'fill_alpha': 1, 'curve_mnems': ['VSH']},
            'curve_category_01c': {'name': 'Open Perforation', 'bin': 'Bin01', 'xmin': 10, 'xmax': 0, 'unit': 'Flag', 'color': 'black', 'fill': True, 'fill_direction': 'left_to_right', 'fill_alpha': 1, 'curve_mnems': ['Perf_open']},
            'curve_category_01d': {'name': 'Closed Perforation', 'bin': 'Bin01', 'xmin': 10, 'xmax': 0, 'unit': 'Flag', 'color': 'red', 'fill': True, 'fill_direction': 'left_to_right', 'fill_alpha': 1, 'curve_mnems': ['Perf_closed']},
            'curve_category_02a': {'name': 'Shallow Resistivity', 'bin': 'Bin02', 'xmin': 0.2, 'xmax': 20000, 'unit': 'ohm.m', 'color': 'grey', 'fill': True, 'fill_direction': 'left_to_right', 'fill_alpha': 0.2, 'curve_mnems': ['RES', 'RESS', 'RESM']},
            'curve_category_02b': {'name': 'Deep Resistivity', 'bin': 'Bin02', 'xmin': 0.2, 'xmax': 20000, 'unit': 'ohm.m', 'color': 'black', 'fill': True, 'fill_direction': 'left_to_right', 'fill_alpha': 0.2, 'curve_mnems': ['RESD']},
            'curve_category_03a': {'name': 'Neutron Porosity', 'bin': 'Bin03', 'xmin': -0.15, 'xmax': 0.45, 'unit': 'fraction', 'color': 'blue','fill': False, 'fill_direction': None, 'fill_alpha': None, 'curve_mnems': ['NPHI', 'NPOR', 'NPSS', 'NPLS', 
                                                                                                                                       'NPDL']},
            'curve_category_03b': {'name': 'Bulk Density', 'bin': 'Bin03', 'xmin': 1.9, 'xmax': 2.65, 'unit': 'g/cc', 'color': 'black','fill': False, 'fill_direction': None, 'fill_alpha': None, 'curve_mnems': ['RHOB']},
            'curve_category_03c': {'name': 'Bit Size', 'bin': 'Bin03', 'xmin': 7, 'xmax': 15, 'unit': 'inches', 'color': 'purple','fill': False, 'fill_direction': None, 'fill_alpha': None, 'curve_mnems': ['BS']},
            'curve_category_03d': {'name': 'Caliper', 'bin': 'Bin03', 'xmin': 7, 'xmax': 15, 'unit': 'inches', 'color': 'orange','fill': False, 'fill_direction': None, 'fill_alpha': None, 'curve_mnems': ['CAL', 'CALI']},
            'curve_category_03e': {'name': 'Density Correction', 'bin': 'Bin03', 'xmin': 0, 'xmax': 1, 'unit': 'g/cc', 'color': 'blue','fill': False, 'fill_direction': None, 'fill_alpha': None, 'curve_mnems': ['DRHO']},
            'curve_category_03f': {'name': 'Density Porosity', 'bin': 'Bin03', 'xmin': -0.15, 'xmax': 0.45, 'unit': 'fraction', 'color': 'red','fill': False, 'fill_direction': None, 'fill_alpha': None, 'curve_mnems': ['PHIDSS_RHOB_CALC', 'PHID', 'PHISS',]},
            'curve_category_04': {'name': 'Water Saturation', 'bin': 'Bin04', 'xmin': 1, 'xmax': 0, 'unit': 'fraction', 'color': 'green', 'fill': True, 'fill_direction': 'right_to_left', 'fill_alpha': 0.5, 'curve_mnems': ['SWE', 'SW']},
            'curve_category_05': {'name': 'Sonic', 'bin': 'Bin05', 'xmin': 50, 'xmax': 200, 'unit': 'us/f', 'color': 'black', 'fill': False, 'fill_direction': None, 'fill_alpha': None, 'curve_mnems': ['DT', 'DTC', 'DTS']},
            # 'curve_category_5': {'name': 'Mud Gases', 'bin': 'Bin5', 'xmin': 1, 'xmax': 10000, 'unit': 'ppm', 'color': 'black', 'fill'=True, 'curve_mnems': ['C1_GAS', 'C2_GAS', 'C3_GAS', 'IC4_GAS', 'IC5_GAS', 
            #                                                                                                          'NC5_GAS', 'TOTALGAS', 'NC4_GAS']},
            'curve_category_06a': {'name': 'C1 Gas', 'bin': 'Bin06', 'xmin': 0.1, 'xmax': 10000, 'unit': 'ppm', 'color': 'red', 'fill':True, 'fill_direction': 'left_to_right', 'fill_alpha': 0.2, 'curve_mnems': ['C1_GAS']},
            'curve_category_06b': {'name': 'C2 Gas', 'bin': 'Bin06', 'xmin': 0.1, 'xmax': 10000, 'unit': 'ppm', 'color': 'green', 'fill':True, 'fill_direction': 'left_to_right', 'fill_alpha': 0.2, 'curve_mnems': ['C2_GAS']},
            'curve_category_06c': {'name': 'C3 Gas', 'bin': 'Bin06', 'xmin': 0.1, 'xmax': 10000, 'unit': 'ppm', 'color': 'turquoise', 'fill':True, 'fill_direction': 'left_to_right', 'fill_alpha': 0.2, 'curve_mnems': ['C3_GAS']},
            'curve_category_06d': {'name': 'iC4 Gas', 'bin': 'Bin06', 'xmin': 0.1, 'xmax': 10000, 'unit': 'ppm', 'color': 'orange', 'fill':True, 'fill_direction': 'left_to_right', 'fill_alpha': 0.2, 'curve_mnems': ['IC4_GAS']},
            'curve_category_06e': {'name': 'nC4 Gas', 'bin': 'Bin06', 'xmin': 0.1, 'xmax': 10000, 'unit': 'ppm', 'color': 'deeppink', 'fill':True, 'fill_direction': 'left_to_right', 'fill_alpha': 0.2, 'curve_mnems': ['NC4_GAS']},
            'curve_category_06f': {'name': 'iC5 Gas', 'bin': 'Bin06', 'xmin': 0.1, 'xmax': 10000, 'unit': 'ppm', 'color': 'darkorchid', 'fill':True, 'fill_direction': 'left_to_right', 'fill_alpha': 0.2, 'curve_mnems': ['IC5_GAS']},
            'curve_category_06g': {'name': 'nC5 Gas', 'bin': 'Bin06', 'xmin': 0.1, 'xmax': 10000, 'unit': 'ppm', 'color': 'blue', 'fill':True, 'fill_direction': 'left_to_right', 'fill_alpha': 0.2, 'curve_mnems': ['NC5_GAS']},
            'curve_category_07': {'name': 'Oil show', 'bin': 'Bin07', 'xmin': 0, 'xmax': 2, 'unit': 'Show Quality', 'color': 'brown', 'fill':True, 'fill_direction': 'left_to_right', 'fill_alpha': 0.5, 'curve_mnems': ['OIL_SHOW', 'OIL SHOW', 'OIL_SHOW_BINARY']},
            'curve_category_08a': {'name': 'Wetness Ratio', 'bin': 'Bin08', 'xmin': 0.1, 'xmax': 200, 'unit': 'frac', 'color': 'blue', 'fill':True, 'fill_direction': 'left_to_right', 'fill_alpha': 0.2, 'curve_mnems': ['WETNESS_RATIO']},
            'curve_category_08b': {'name': 'Balance Ratio', 'bin': 'Bin08', 'xmin': 0.1, 'xmax': 200, 'unit': 'frac', 'color': 'red', 'fill':True, 'fill_direction': 'left_to_right', 'fill_alpha': 0.2, 'curve_mnems': ['BALANCE_RATIO']},
            'curve_category_08c': {'name': 'Character Ratio', 'bin': 'Bin08', 'xmin': 0.1, 'xmax': 200, 'unit': 'frac', 'color': 'green', 'fill':True, 'fill_direction': 'left_to_right', 'fill_alpha': 0.2, 'curve_mnems': ['CHARACTER_RATIO']},
            'curve_category_09': {'name': 'Logistic Regression_Predictions', 'bin': 'Bin09', 'xmin': 0, 'xmax': 2, 'unit': 'Show Quality', 'color': 'black', 'fill':True, 'fill_direction': 'left_to_right', 'fill_alpha': 0.2, 'curve_mnems': ['Logistic Regression_Predictions']},
            'curve_category_10': {'name': 'Random Forest Classifier_Predictions', 'bin': 'Bin10', 'xmin': 0, 'xmax': 2, 'unit': 'Show Quality', 'color': 'black', 'fill':True, 'fill_direction': 'left_to_right', 'fill_alpha': 0.6, 'curve_mnems': ['Random Forest Classifier_Predictions']},
            'curve_category_11': {'name': 'Gradient Boosting Classifier_Predictions', 'bin': 'Bin11', 'xmin': 0, 'xmax': 2, 'unit': 'Show Quality', 'color': 'black', 'fill':True, 'fill_direction': 'left_to_right','fill_alpha': 0.6, 'curve_mnems': ['Gradient Boosting Classifier_Predictions']},
            'curve_category_12': {'name': 'XGBoost Classifier_Predictions', 'bin': 'Bin12', 'xmin': 0, 'xmax': 2, 'unit': 'Show Quality', 'color': 'black', 'fill':True, 'fill_direction': 'left_to_right','fill_alpha': 0.6, 'curve_mnems': ['XGBoost Classifier_Predictions']},
            'curve_category_13': {'name': 'Neural Network Classifier_Predictions', 'bin': 'Bin13', 'xmin': 0, 'xmax': 2, 'unit': 'Show Quality', 'color': 'black', 'fill':True, 'fill_direction': 'left_to_right','fill_alpha': 0.6, 'curve_mnems': ['Neural Network Classifier_Predictions']},
            'curve_category_14': {'name': 'SVM Classifier_Predictions', 'bin': 'Bin14', 'xmin': 0, 'xmax': 2, 'unit': 'Show Quality', 'color': 'black', 'fill':True, 'fill_direction': 'left_to_right','fill_alpha': 0.6, 'curve_mnems': ['SVM Classifier_Predictions']},
            'curve_category_15': {'name': 'KNN Classifier_Predictions', 'bin': 'Bin15', 'xmin': 0, 'xmax': 2, 'unit': 'Show Quality', 'color': 'black', 'fill':True, 'fill_direction': 'left_to_right','fill_alpha': 0.6, 'curve_mnems': ['KNN Classifier_Predictions']},
            'curve_category_16': {'name': 'C2 Oil Show Indicator', 'bin': 'Bin16', 'xmin': 0, 'xmax': 2, 'unit': 'Show Quality', 'color': 'black', 'fill':True, 'fill_direction': 'left_to_right','fill_alpha': 0.6, 'curve_mnems': ['C2 Oil Show Indicator']},
            'curve_category_17': {'name': 'Gradational Flag', 'bin': 'Bin17', 'xmin': 0, 'xmax': 1, 'unit': 'Flag', 'color': 'white', 'fill':True, 'fill_direction': 'left_to_right', 'fill_alpha': None, 'curve_mnems': ['Gradational_flag']},
            'curve_category_18': {'name': 'Generic Flag', 'bin': 'Bin18', 'xmin': 0, 'xmax': 2, 'unit': 'Flag', 'color': 'black', 'fill':True, 'fill_direction': 'left_to_right', 'fill_alpha': 0.5, 'curve_mnems': ['Flag']},
            'curve_category_19': {'name': 'Kmeans Cluster Label', 'bin': 'Bin19', 'xmin': 0, 'xmax': 1, 'unit': 'Flag', 'color': 'black', 'fill':True, 'fill_direction': 'left_to_right', 'fill_alpha': None, 'curve_mnems': ['KMeans_Category', 'Cluster_label', 'Category_code']},
            'curve_category_97': {'name': 'RHOB Rate Change', 'bin': 'Bin99', 'xmin': -0.5, 'xmax': 0.5, 'unit': 'g/(cc*ft)', 'color': 'black', 'fill':False, 'fill_direction': 'left_to_right', 'fill_alpha': None, 'curve_mnems': ['RHOB_rate']},
            'curve_category_98': {'name': 'ResD Rate Change', 'bin': 'Bin98', 'xmin': -0.5, 'xmax': 0.5, 'unit': 'ohm.m/ft', 'color': 'black', 'fill':False, 'fill_direction': 'left_to_right','fill_alpha': None, 'curve_mnems': ['RESD_LOG10_rate']},
            'curve_category_99': {'name': 'ResD / RHOB Rate Change', 'bin': 'Bin97', 'xmin': -500, 'xmax': 500, 'unit': 'ohm.m/ft / g/(cc*ft)', 'color': 'black', 'fill':False, 'fill_direction': 'left_to_right', 'fill_alpha': None, 'curve_mnems': ['dRhoB_div_dRes10_LOG10']},
            }
# The important details regarding each curve, including x scale (log or lin),
# what the width of the track should be, what the name of the curve track
# should be etc.
bin_params = {'Bin01': {'name': 'Correlation', 'width': 10, 'x_scale': 'linear'},
              'Bin02': {'name': 'Resistivity', 'width': 10, 'x_scale': 'log',},
              'Bin03': {'name': 'Porosity', 'width': 10, 'x_scale': 'linear',},
              'Bin04': {'name': 'Water Saturation', 'width': 10, 'x_scale': 'linear',},
              'Bin05': {'name': 'Sonic', 'width': 10, 'x_scale': 'linear',},
              'Bin06': {'name': 'Mud Gases', 'width': 10, 'x_scale': 'log',},
              'Bin07': {'name': 'Oil Show', 'width': 10, 'x_scale': 'linear',},
              'Bin08': {'name': 'Mud Gas Ratios', 'width': 10, 'x_scale': 'log',},
              'Bin09': {'name': 'Logistic Regression_Predictions', 'width': 10, 'x_scale': 'linear',},
              'Bin10': {'name': 'Random Forest Classifier_Predictions', 'width': 10, 'x_scale': 'linear',},
              'Bin11': {'name': 'Gradient Boosting Classifier_Predictions', 'width': 10, 'x_scale': 'linear',},
              'Bin12': {'name': 'XGBoost Classifier_Predictions', 'width': 10, 'x_scale': 'linear',},
              'Bin13': {'name': 'Neural Network Classifier_Predictions', 'width': 10, 'x_scale': 'linear',},
              'Bin14': {'name': 'SVM Classifier_Predictions', 'width': 10, 'x_scale': 'linear',},
              'Bin15': {'name': 'KNN Classifier_Predictions', 'width': 10, 'x_scale': 'linear',},
              'Bin16': {'name': 'C2 Pay Indicator', 'width': 10, 'x_scale': 'linear',},
              'Bin17': {'name': 'Gradational Flag', 'width': 10, 'x_scale': 'linear',},
              'Bin18': {'name': 'Generic Flag', 'width': 10, 'x_scale': 'linear',},
              'Bin19': {'name': 'Kmeans Cluster Label', 'width': 10, 'x_scale': 'linear',},
              'Bin97': {'name': 'RHOB Rate Change', 'width': 10, 'x_scale': 'linear',},
              'Bin98': {'name': 'RESD Rate Change', 'width': 10, 'x_scale': 'linear',},
              'Bin99': {'name': 'ResD / RHOB Rate Change', 'width': 10, 'x_scale': 'linear',}
              }

LAS_load_error_log = []

bin_counter_dict = {}


class Curve():
    #TODO Use the curve object as a portable way to move around data associated
    #with log curves    
    
    #TODO Think about Curves as a child class of log... how can we inherit
    #the attributes of the Log object in this child class?
    def __init__(self, curve_name, curve_data):
        
        curve_category, curve_bin = get_category_and_bin_for_curve(curve_name)
        
        curve_params_dict = curve_to_category_dict[curve_category]
        
        curve_scale_type = bin_params[curve_bin]['x_scale']
        
       
        self.name = curve_name
        self.color = curve_params_dict['color']
        self.fill = curve_params_dict['fill']
        self.fill_direction = curve_params_dict['fill_direction']
        self.fill_alpha = curve_params_dict['fill_alpha']
        self.category_name = curve_params_dict['name']
        self.units = curve_params_dict['unit']
        self.scale_min = curve_params_dict['xmin']
        self.scale_max = curve_params_dict['xmax']
        self.curve_scale = curve_scale_type
        
        
        self.data = curve_data


def get_category_and_bin_for_curve(curve_name):
    """
    Look through curve category dict for a match for curve name. Return curve
    category and bin name associated with that particular curve

    Parameters
    ----------
    curve_name : str
        The mnemonic of the curve.

    Returns
    -------
    category_for_curve : str
        The category corresponding to the curve.
    bin_for_curve : TYPE
        The bin or track corresponding to the curve.

    """
    
    #Loop over the curve to category dict     
    for curve_category, sub_dict in curve_to_category_dict.items():
        
        #Curve mnems for category are the curve mnemonics that alias to the
        #curve category
        curve_mnenems_for_category = sub_dict['curve_mnems']
        
        # If the value of the dictionary item 'curve_mnems' is a list, and the curve
        # is in that list, append a tuple of the curve name and the
        # associated bin number to the plot_curves_name list.
        
        #Start with try except block to see if the curve mnemonic is in 
        #the list of mnemonics that correspond to that curve category
        try:
            if (isinstance(curve_mnenems_for_category, list) and 
                curve_name in curve_mnenems_for_category):
                
                category_for_curve = curve_category
                
                bin_for_curve = sub_dict['bin']
                
                                   
        except KeyError:
            print(f'The curve {curve_name} did not match')
            LAS_load_error_log.append((well_name, curve_name))
            continue
        
        #TODO Create more robust exception handling for exceptions other
        #than key error
        
        except:
            print('An unknown error was encountered')
            LAS_load_error_log.append((well_name, 'unknown'))
            continue
    
    return (category_for_curve, bin_for_curve)

def depth_index_log_data(curve_data_df):
    """
    Take a dataframe of ASCII log data with a multiindex, and set the
    dataframe index to depth only

    Parameters
    ----------
    curve_data_df : DataFrame
        Dataframe with ASCII log data from log object.

    Returns
    -------
    curve_data_df : DataFrame
        The input dataframe with only the depth as the index.

    """
    
    #TODO Create a filter to make sure the correct data type is passed into this function
    
    #Drop the well name from the multiindex. Assume that depth is level 1;
    #normally the well name is level 0 and the depth is level 1.
    curve_data_df = curve_data_df.reset_index()
    
    #The depth is now the only index. Put a depth column back into the 
    #dataframe
    curve_data_df.index = curve_data_df['DEPT'] 
    
    return curve_data_df


def get_unique_entries(item_list):
    """Return the list of unique items in a list"""
    item_list_set = set(item_list)
    list_deduplicated = list(item_list_set)
    list_deduplicated.sort()
    
    return list_deduplicated


def get_curve_data(curve_name, log_data):
    """Slice out the data for a particular curve from an ascii log data dataframe"""
    
    #Return all rows ":" corresponding to the column 'curve name'
    
    curve_data = log_data.loc[:,curve_name]
    #curve_data.index = curve_depth
    
    return curve_data
    

def consolidate_bin_and_curves(list_of_bins_and_curves, log_data):
    
    #Unpack the tuples of bin, curve pairs in list_of_all_bins_and_curves
    all_bin_entries, all_curve_entries = zip(*list_of_bins_and_curves)
    
    #Get names of unique bins in all_bin_entries list
    unique_bins = get_unique_entries(all_bin_entries)
    
    #Instantiate a dictionary that will create an empty list associated with
    #each unique bin name
    
    consolidated_data_dict = {unique_bin: [] for unique_bin in unique_bins}
    
    #Loop over the list that has the bin-curve tuples
    for (bin_name, curve_name) in list_of_bins_and_curves:
        
        #Take the log_data data frame and pull out the data that is only
        #associated with that mneumonic
        curve_data = get_curve_data(curve_name, log_data)
        
        #Create a Curve object with the curve name and the curve data
        curve_obj = Curve(curve_name, curve_data)
        
        #Add Curve object to the list under the corresponding bin in the
        #consolidated_data_dict dictionary
        
        consolidated_data_dict[bin_name].append(curve_obj)
    
    for curve_list in consolidated_data_dict.items():
        curve_list = [(curve, index) for index, curve in enumerate(curve_list)]
            
    
    return consolidated_data_dict
        
def get_validated_curves(log_data):
    """Check the list of curve mnemonics in the log object against the list
    of curve mneumonics in the curve_to_category dictionary. Return a dictionary
    with the bins to populate as keys, and a list of curves within those bins
    as the values"""
    bin_and_curves_list = []
    
    #Prepare to iterate over log data dataframe column headers
    for curve_name in log_data.keys():
        
        #Prepare to iterate over curve categories and associated mnemonics
        for curve_category, sub_dict in curve_to_category_dict.items():
            
          
            #curve_mnenems_for_category is the list of curves associated 
            #with a particular category            
            curve_mnenems_for_category = sub_dict['curve_mnems']
            
            # If the value of the dictionary item 'curve_mnems' is a list, and the curve
            # is in that list, append a tuple of the curve name and the
            # associated bin number to the plot_curves_name list.
            
            #Start with try except block to see if the curve is in the list
            try:
                if (isinstance(curve_mnenems_for_category, list) and 
                    curve_name in curve_mnenems_for_category):
                    
                    bin_for_curve = sub_dict['bin']
                    
                    bin_and_curves_list.append((bin_for_curve, curve_name))
                    
                    break
            
            except KeyError as key_error:
                print(f'The curve {curve_name} did not match')
                LAS_load_error_log.append((well_name, curve_name))
                continue
            
            #TODO Create more robust exception handling for exceptions other
            #than key error
            
            except:
                print('An unknown error was encountered')
                LAS_load_error_log.append((well_name, 'unknown'))
                continue
            
    consolidated_bin_and_curves_dict = consolidate_bin_and_curves(bin_and_curves_list, log_data)
    
    consolidated_bin_and_curves_list = list(consolidated_bin_and_curves_dict.items())
    
    return consolidated_bin_and_curves_list

def create_subplots(validated_curves):
    #Calculate the number of tracks or subplots that will be created, equal
    #to the length of the validated_curves return. The return is equal to the
    #number of bins that will have data
    num_tracks = len(validated_curves)
    
    #Instantiate dictionary to store axes objects
    axs_dict = {}
    
    #Counter to dynamically name axes objects
    axs_counter = 1
    
    #Loop over dictionary of bin and curves
    for index, (bin_name, curves) in enumerate(validated_curves):
        
        #The number of curves in the track is equal to the length of the curve
        #list associated with the bin
        num_curves_in_track = len(curves)
        
        #Create a string that will be set as the key value for the dictionary
        #entry
        axs_indexer = f'ax{axs_counter}'
        
        #Instantiate the axes object for the bin, and include information
        #on which track has been created, and that this is the first curve in
        #the track
        axs_dict[axs_indexer] = (plt.subplot2grid((1, num_tracks), (0, index)), dict(Track=index, Position_in_track=1))
        
        #Increment axes counter so that next axes is unique (eg. ax2 will come
        #after ax1)
        axs_counter += 1
        
        #If there is more than one curve associated with the track, for each
        #curve after the first, duplicate the first axes object for that track
        #and store it in the dictionary and store the associated track and
        #curve information for that axes object, as above
        if num_curves_in_track > 1:
            for sub_index, curve, in enumerate(curves[1:]):
                
                #This is plus 2 because the difference between the index and the track number is always 1,
                #and the slice starts at the second entry (index 1), so the first position_in_track we are looking
                #at in this loop is the second curve in the track
                position_in_track = sub_index + 2 
                
                #Duplicate the first axes for that track and store the
                #associated information for that axes in a sub-dictionary, as
                #above
                axs_dict[f'ax{axs_counter}'] = (axs_dict[axs_indexer][0].twiny(), dict(Track=index, Position_in_track=position_in_track))
                
                #Increment axes counter
                axs_counter += 1
    
    return axs_dict

#TODO Think about how I can make curve fills into their own class

def constant_curve_fill(ax, curve, x_data, y_data, x_max, color, alpha_value):
    """Assign a constant curve fill"""
    
    plt.sca(ax)
    if curve.fill_direction == 'left_to_right':
        plt.fill_betweenx(y_data, x_data, color=color, alpha=alpha_value)
    if curve.fill_direction == 'right_to_left':
        plt.fill_betweenx(y_data, x_data, 200, color=color)

def variable_curve_fill(ax, curve, x_data, y_data, x_min, x_max, **kwargs):
    """Assign a variable curve fill using a colormap"""
    #Ensure current axes is set to ax
    plt.sca(ax)

    #Assign curve name to variable
    curve_name = curve.name
    
    #Plot the variale fill. This is only set up for GR right now but could be
    #extended to other curves. 
    if curve_name == 'GR':
        
        #Span is the difference between the maximum and minimum value
        span = x_max - x_min
        
        #Get the colour map
        color_map = plt.get_cmap('viridis_r') 
        
        #Create the range of values to iterate over when drawing the curve fill.
        #the denominator in the last argumnent will determine how many different
        #colours appear in the color map, therefy affecting the "fineness" of
        #the colour scale
        
        color_index = np.arange(x_min, x_max, span / 100)
    
        #Iterate over the values in the colour index. Normalize the value back
        #to a scale between 0 and 1 in order to retrieve the associated colour
        #and assign to the index_value variable for that iteration. Retrieve
        #the colour associated with the 0 to 1 value. Then find all values where
        #the x values are greater than the specified value and draw the color.
        #Repeat for all values in the colour index.
        
        for index in sorted(color_index):
            index_value = (index - x_min) / span
            color = color_map(index_value)
            plt.fill_betweenx(y_data, x_data, 200, where = x_data >= index, color=color)
            
    cluster_curve_names = ['KMeans_Category', 'Cluster_label', 'Category_code']
    if curve_name in cluster_curve_names:
                        
       
        #Extract df
        curve_data_df = pd.DataFrame(curve.data)
                
        #codes = kwargs.get('category_codes', None)
        
        if curve_name != 'Category_code':
       
            #Create categorical object
            categories = pd.Categorical(curve_data_df[curve_name])
            
            #Create codes column, assigning an integer to every unique value
            codes = categories.codes
                   
            curve_data_df['Category_code'] = codes
            
            codes = pd.Series(codes)
        else:
            
            codes = curve_data_df['Category_code']
        
        codes_no_nan = codes.dropna()
        
        unique_codes = list(set(codes_no_nan.to_list()))
        
        norm = kwargs.get('norm', None)
        
        if norm is None:
            min_cat_value = curve_data_df['Category_code'].min()
            max_cat_value = curve_data_df['Category_code'].max()
            norm = mcolors.Normalize(vmin=min_cat_value, vmax=max_cat_value)
        
        category_color_map = cmap.get_cmap('tab10') #Tableau 10 colormap
        
        
        
      
        for code in unique_codes:
             color = category_color_map(norm(code))
             
             condition = curve_data_df['Category_code'] == code
            
             plt.fill_betweenx(y_data, -1, x_data, where=condition, color=color)
    
    #TODO Finish this function      
    if curve_name == 'Gradational_flag':
        # Create a custom colormap
        colors = [(0, 0, 0, 0), (0, 0, 0, 1)]  # From black (transparent) to black (opaque) 
        n_bins = 100  # Number of color steps in the colormap
        cmap_name = 'black_to_transparent'
        colormap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
        
        color_index = np.arange(0,1,(1/n_bins))
        
        rounded_x_data = [((round(i *100)) / 100) for i in x_data]
        

        
        for index in color_index:
            color = colormap(index)
            
            plt.fill_betweenx(y_data, 0, 1, where = rounded_x_data == index, color=color)
        
        
                
    else:
        pass
        
    
def plot_curves(fig, 
                axs_dict, 
                validated_curves, 
                curve_depth_limits, 
                well_name, 
                save_fig, **kwargs):
    """Plotting function for the curves"""
    
    #Unpack **kwargs values, otherwise establish default values
    
    major_tick_interval = kwargs.get('major_tick_interval', 500)
    minor_tick_interval = kwargs.get('minor_tick_interval', 100)
    plot_comment = kwargs.get('comment', None) #To leave a comment at the bottom right of the plot
    
    
    
    
    #Unpack validated curves into separate lists
    bins, curve_lists = zip(*validated_curves)
    
    #Flatten the list of curves associated with each bin
    flattened_curve_list = [curve for sub_list in curve_lists for curve in sub_list]
    
    #Use list comprehension to get all axes from dictionary into a list 
    flattened_axs_list = [entry[1][0] for entry in axs_dict.items()]
    
    #Get a list of The position of each curve in each track
    flattened_position_list = [entry[1][1]['Position_in_track'] for entry in axs_dict.items()]
    
    #Zip the lists of curves, the respective axes of the curve, and the position
    #of the curve in each track in preparattion to "iterate over the axes"
    axs_curve_tuple_list = list(zip(flattened_axs_list, flattened_curve_list,
                                    flattened_position_list))
    
   
    for (ax, curve, position) in axs_curve_tuple_list:
        plt.sca(ax)
        
        #Set variables that will be used for plotting
        # print (f'Plotting {ax}')
        curve_name = curve.name
        x_data = list(curve.data.values)
        
        y_data = list(curve.data.index.to_list())
        curve_data = (x_data, y_data)
        curve_color = curve.color
        line_width = 0.5
        fill_alpha_value = curve.fill_alpha
        x_min =  curve.scale_min
        x_max = curve.scale_max
        y_min = curve_depth_limits[0]
        y_max = curve_depth_limits[1]
        x_curve_scale_type = curve.curve_scale
                
        #Main plotting call for curves
        if curve_name == 'Gradational_flag': #Plot gradational flag line as straight line
            x_data_ones = [1 for i in x_data]
            ax.plot(x_data_ones, y_data, color='none', lw=line_width)
            
        else:
        
            ax.plot(x_data, y_data, color=curve_color, lw=line_width)
            
        
        #Set major and minor tick parameters
        ax.yaxis.set_major_locator(MultipleLocator(major_tick_interval))
        ax.yaxis.set_minor_locator(MultipleLocator(minor_tick_interval))
        
        #Remove labels from subplots
        ax.tick_params(axis='y', labelleft=False) ##NEW 2
            
                
        if curve.fill == True:
            curve_fill_names = ['GR', 'Gradational_flag', 'Cluster_label', 'Category_code']
            
            if curve_name in curve_fill_names:
                
                variable_curve_fill(ax, curve, x_data, y_data, x_min, x_max, **kwargs)
            else:
                constant_curve_fill(ax, curve, x_data, y_data, x_max, curve_color, fill_alpha_value)
                
              
                
        
        #Set depth limits for plot
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min) #This is done in reverse order to show a proper "depth" display
        
        #Add grid
        
        #For the first curve in each track
        if position == 1:
            ax.xaxis.set_ticks_position("top")
            ax.xaxis.set_label_position("top")

            #Set label
            ax.set_xlabel(curve_name)

            #Set scale colour to the curve colour
            ax.spines['top'].set_color(curve_color)
            
            #Draw grid for the first curve with major and minor increments
            ax.grid(True, which='major', axis='y', color='gray', linestyle='-', linewidth=1)  # Major grid
            ax.grid(True, which='minor', axis='y', color='gray', linestyle='--', linewidth=0.7, alpha=0.3)  # Minor grid, transparent
            ax.grid(True, which='both', axis='x')
            
            
        
        #For each subsequent curve
        else:
            ax.xaxis.set_ticks_position("top")
            ax.xaxis.set_label_position("top")
            ax.set_xlabel(curve_name)
            
            #Offset the next scales to be on top of first scale.
            
            offset = 0.985 + (0.02*position)
            ax.spines["top"].set_position(("axes", offset))
            
            #Set scale colour to the curve colour
            ax.spines['top'].set_color(curve_color)
        
        if x_curve_scale_type == 'log':
            ax.semilogx()
        else:
            pass
        



 
    #plt.show()
    
    # tick_intervals = (major_tick_interval, minor_tick_interval)
    
    return fig, ax, axs_dict
        
        

    
    

def create_plot(well_log, 
                well_name=None, 
                min_depth=None, 
                max_depth=None, 
                save_fig=False, **kwargs):
    """
    This is the main function to create well log plots.

    Parameters
    ----------
    well_log : Log object
        The log object that contains the data to be plotted.
    well_log_limits : TYPE
        DESCRIPTION.
    well_name : TYPE
        DESCRIPTION.
        
    min_depth : str or float
        This can be a formation top in string format, or a specific MD depth
    
    max_depth : str or float
        This can be a formation top in string format, or a specific MD depth

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.
    axs_dict : TYPE
        DESCRIPTION.

    """
    
    if well_name == None:
        try:
            well_name = well_log.name
        except AttributeError as a:
            print(a)
            print('Error assigning name')
    
    #Get ascii data from log object
    well_log_data = well_log.A
    
    #Drop columns with all nans
    well_log_data = well_log_data.dropna(axis=1, how='all')
    
    #Establish curve limits for log
    well_log_limits = well_log.curve_limits
    
    #Clean up and reindex the ascii data dataframe
    conditioned_curve_data = depth_index_log_data(well_log_data)
    
    well_log_data = conditioned_curve_data
    
   
    #Return the curves in the ascii dataframe that are pre-identified
    validated_curves = get_validated_curves(well_log_data)
    
    #Instantiate the figure object
    fig, axs = plt.subplots(figsize = (10,40), sharey=True)
    
    axs.set_xticks([])
    # axs.set_yticks([])    
        
    #Tight layout for the plot
    plt.tight_layout()
    
    #Call to create subplots based on the bins that were identified in
    #validated_curves(). Return 
    axs_dict = create_subplots(validated_curves)
    
    #Check if the optional parameters for depth were passed. If not, use defaults
    #from well_log_limits tuple
    if min_depth == None:
        min_depth = well_log_limits[0]
        
    if max_depth == None:
        max_depth = well_log_limits[1]
    
        
    #Use curve_depth limits to find the curve depth limits. Set the limits
    #to the well maximum and minimum if a top is not found
    try:
        curve_depth_limits = (min_depth, max_depth)
    except ValueError:
        print('One of the specified tops was not found in the data')
        min_depth = well_log_limits[0]
        max_depth = well_log_limits[1]
        
    #Call the plot curves function to plot the data
    fig, ax, axs_dict = plot_curves(fig, axs_dict, validated_curves, curve_depth_limits, well_name, save_fig, **kwargs)
    
    fig.suptitle(well_name, x=0, y=0, fontsize=16)
    
    
    #Save figure, if specified
    if save_fig == True:
        save_fig_fn = f'{well_name}_plot.png'
        fig.savefig(save_fig_fn, bbox_inches='tight')
    
    plt.show()
    
    return fig, ax, axs_dict


        

def get_category_codes(df, cluster_label='Cluster_label'):
    """
    Get the integer category codes to normalize the color scale, and modify
    the dataframe to include a column with the integer category code for
    consistent assignment of colors

    Parameters
    ----------
    df : DataFrame
        The dataframe containing the data to be plotted.
    cluster_label : str, optional
        DESCRIPTION. The column index for the column with the raw cluster labels

    Returns
    -------
    df : DataFrame
        The modified data frame with the integer cluster code column.
    codes : Series
        A series just with the integer code data as values.

    """
    
    
    cluster_series = df[cluster_label]
    #Create categorical object
    categories = pd.Categorical(cluster_series)
    
    #Create codes column, assigning an integer to every unique value
    codes = categories.codes
    
    df['Category_code'] = codes
    
    return (df, codes)



def get_norm_object(cluster_series):
    """
    Create a normalization object that will allow the same color mapping to
    propogate across multiple plots. This can be passed as a keyword argument

    Parameters
    ----------
    cluster_series : Series
        The series corresponding to the cluster labels.

    Returns
    -------
    norm : normalization object
        Matplotlib normalization object.

    """
    
    min_cat_value = cluster_series.min()
    max_cat_value = cluster_series.max()
    
    norm = mcolors.Normalize(vmin=min_cat_value, vmax=max_cat_value)
    
    return norm



def create_plots_with_df_and_megadf(sub_df, mega_df, save_fig=False, min_depth=None, max_depth=None, **kwargs):
    """
    Call create plot with all the wells mentioned in the sub_df with all the data from mega_df
    
    Parameters
    ----------
    sub_df : DataFrame
        A subset of mega_df. This may have been derived through different means
        including filtering.
        
    mega_df : DataFrame
        The main, consolidated source of curve data
        
    min_depth : str or float
        This can be a formation top in string format, or a specific MD depth
    
    max_depth : str or float
        This can be a formation top in string format, or a specific MD depth

    Returns
    -------
    plots_dict : dictionary
        A dictionary indexed by well name with the fig, ax and axes_dict objects for each well 

    """
    
    keywords = {}
    
    plots_dict = {}
    
    well_names = list(set(list(sub_df.index.get_level_values('Well_name'))))
    
    if 'Cluster_label' in sub_df.columns:
        sub_df, codes = get_category_codes(sub_df)

        codes_series = sub_df['Category_code']
                
        mega_df = pd.concat([mega_df, codes_series], axis=1) #Merge category codes to mega_df
        norm = get_norm_object(codes)
        
        
        keywords['norm'] = norm
    
    for well_name in well_names:
        df_temp = mega_df[mega_df.index.get_level_values('Well_name') == well_name]
        fig, ax, axs_dict = create_plot_with_df(df_temp, well_name, save_fig=save_fig,
                            min_depth=min_depth, max_depth=max_depth, **keywords)
        plots_dict[well_name] = (fig, ax, axs_dict)
        
    return plots_dict   


def create_plots_with_df(df, save_fig=False, min_depth=None, max_depth=None, **kwargs):
    
    
    """
    Call create plot with df with all the wells mentioned in the dataframe
    

    Parameters
    ----------
    df : DataFrame
        Curve data dataframe.

    Returns
    -------
    plots_dict : dictionary
        A dictionary indexed by well name with the fig, ax and axes_dict objects for each well 

    """
    
    keywords = {}
    
    plots_dict = {}
    
    well_names = list(set(list(df.index.get_level_values('Well_name'))))
    
    if 'Cluster_label' in df.columns:
        df, codes = get_category_codes(df)
        norm = get_norm_object(codes)
        
        
        keywords['norm'] = norm
            
    for well_name in well_names:
        df_temp = df[df.index.get_level_values('Well_name') == well_name]
        fig, ax, axs_dict = create_plot_with_df(df_temp, well_name, save_fig=save_fig,
                            min_depth=min_depth, max_depth=max_depth, **keywords)
        plots_dict[well_name] = (fig, ax, axs_dict)
        
    return plots_dict

def create_plot_with_df(well_log_data,
                        well_name=None,
                        min_depth=None,
                        max_depth=None,
                        save_fig=False,
                        **kwargs):
    """
    This is a modified version of create_plot to accept a dataframe instead of
    a well log object

    Parameters
    ----------
    well_log_data : Dataframe
        The dataframe that contains the ASCII data to be plotted.
    well_log_limits : TYPE
        DESCRIPTION.
    well_name : str, optional
        The well name associated with the dataframe.
    min_depth : str or float
        This can be a formation top in string format, or a specific MD depth
    max_depth : str or float
        This can be a formation top in string format, or a specific MD dept

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.
    axs_dict : TYPE
        DESCRIPTION.

    """
    
    #If well name not specified, index the first record of the 'Well_name'
    #level in the index
    try:
        if well_name == None:
            well_name = well_log_data.index.get_level_values('Well_name')[0]
    except KeyError as e:
        print('Key Error: No entry found for well name')
    
    DEPTH_PAD = 100
    FORMATION_COLUMN = 'Strat_unit_name'
    
    major_tick_interval = kwargs.get('major_tick_interval', 500)
    minor_tick_interval = kwargs.get('minor_tick_interval', 100)
    
    #Drop columns with all nans
    well_log_data = well_log_data.dropna(axis=1, how='all')
    
    
    #Establish curve limits for log. Try to get it from the multi-index,
    #then try to index from the depth column if there is an index error
    try:
        upper_depth_limit = well_log_data.index.get_level_values(1).min()
        lower_depth_limit = well_log_data.index.get_level_values(1).max()
    except IndexError:
        upper_depth_limit = well_log_data['DEPT'].min()
        lower_depth_limit = well_log_data['DEPT'].max()
    
    well_log_limits = (upper_depth_limit, lower_depth_limit)
    
    #Clean up and reindex the ascii data dataframe
    conditioned_curve_data = depth_index_log_data(well_log_data)
    
    well_log_data = conditioned_curve_data
    
        
    #Return the curves in the ascii dataframe that are pre-identified
    validated_curves = get_validated_curves(well_log_data)
    
    #Instantiate the figure object
    fig, axs = plt.subplots(figsize = (10,40))
    
    axs.set_xticks([])
    axs.set_yticks([])
    
    #Tight layout for the plot
    plt.tight_layout()
    
    #Call to create subplots based on the bins that were identified in
    #validated_curves(). Return 
    axs_dict = create_subplots(validated_curves)
        
        
    #Check if the optional parameters for depth were passed. If not, use defaults
    #from well_log_limits tuple
    if min_depth == None: #No min depth specified, so use the shallowest well log limit
        min_depth = well_log_limits[0]
    
    #If a string is passed as min depth, find the shallowest depth corresponding
    #to that formation name and pass it as min_depth
    
    else:
        if type(min_depth) == str:  
            min_depth_label = min_depth
            try:
                df_temp = well_log_data[well_log_data['Strat_unit_name'] == min_depth_label]
                min_depth = df_temp.index.get_level_values('DEPT').min()
                print(f'Min depth assigned from well top {min_depth_label}')
            except IndexError:
                print('Minimum top name did not match')
                min_depth = well_log_limits[0]
        
        #If the dataframe is empty, assign the well log limit to min_depth
        if df_temp.empty:
            min_depth = well_log_limits[0]
            
        
    if max_depth == None:
        max_depth = well_log_limits[1]
    else:
        if type(max_depth) == str:  
            max_depth_label = max_depth
            try:
                df_temp = well_log_data[well_log_data['Strat_unit_name'] == max_depth_label]
                max_depth = df_temp.index.get_level_values('DEPT').max()
                print(f'Max depth assigned from well top {max_depth_label}')
            except IndexError:
                print('Minimum top name did not match')
                max_depth = well_log_limits[1]
        
        #If the dataframe is empty, assign the well log limit to max_depth
        if df_temp.empty:
            max_depth = well_log_limits[1]
        
    #Use curve_depth limits to find the curve depth limits. Set the limits
    #to the well maximum and minimum if a top is not found
    try:
        
        curve_depth_limits = (min_depth, max_depth)
    except ValueError:
        print('One of the specified tops was not found in the data')
        min_depth = well_log_limits[0]
        max_depth = well_log_limits[1]
    
       
    #Call the plot curves function to plot the data
    fig, ax, axs_dict = plot_curves(fig, axs_dict, validated_curves, curve_depth_limits, well_name, save_fig, **kwargs) ##NEW 7
    # fig, ax, axs_dict, tick_intervals = plot_curves(fig, axs_dict, validated_curves, curve_depth_limits, well_name, save_fig) ##NEW 7
    
    fig.suptitle(well_name, x=0, y=0, fontsize=16)
    
    
    # Add a new hidden axis to the left for overall y-axis ticks
    left_ax = fig.add_subplot(111, frameon=False)
    left_ax.set_ylim([max_depth, min_depth])
    
    #Add title to hidden axes
    left_ax.set_xlabel("MD /n Feet")
    
    # Set up tick positions and enable only left-side labels for this axis
    left_ax.yaxis.set_major_locator(MultipleLocator(major_tick_interval))
    left_ax.yaxis.set_minor_locator(MultipleLocator(minor_tick_interval))# Auto-determine tick intervals
    left_ax.tick_params(axis='y', which='both', length=5, labelleft=True, left=True)
    
    # Hide all elements from the fake subplot except y-ticks
    left_ax.tick_params(labeltop=False, labelright=False)  # Hide top and right tick labels
    left_ax.xaxis.set_visible(False)  # Hide x-axis
    left_ax.spines['top'].set_color('none')  # Hide top spine
    left_ax.spines['right'].set_color('none')  # Hide right spine
    left_ax.spines['bottom'].set_color('none')  # Hide bottom spine
    
    # fig.subplots_adjust(left=0.1) ##NEW 4
    # fig.add_subplot(111, frameon=False) ##NEW 4
    # plt.tick_params(labelcolor='none', top=False, bottom=False, left=True, right=False) ##NEW 4
    # plt.grid(False) ##NEW 4
    
    # major_tick_interval, minor_tick_interval = tick_intervals ##NEW 8
    
    # plt.gca().yaxis.set_major_locator(MultipleLocator(major_tick_interval)) ##NEW 5
    # plt.gca().yaxis.set_minor_locator(MultipleLocator(minor_tick_interval)) ##NEW 5
    
    # plt.gca().tick_params(which='minor', length=4, color='r') ##NEW 6
    # plt.gca().tick_params(which='major', length=5, color='r') ##NEW 6
    
    #Save figure, if specified
    if save_fig == True:

        print(f'SaveFig {well_name}')
        save_fig_fn = f'{well_name}_plot.png'
        fig.savefig(save_fig_fn, bbox_inches='tight')
    
    plt.show()
    
    return fig, ax, axs_dict


def create_plot_filter_name_fr_df(multi_well_df,
                                  well_name,
                                  min_depth=None,
                                  max_depth=None,
                                  save_fig=False, **kwargs):
    
    """
    This function allows the user to create a single well plot from a 
    multi-well DataFrame (e.g. mega_df) by specifying the well name
    """
    
    #Filter the multi-well dataframe to a single well
    well_df = meta_plastic.LogDataFilter.well_name_filter(multi_well_df, [well_name], exact_match=True)
    
    #If the dataframe is empty, print the error message
    if well_df.empty:
        print('The dataframe is empty. Check the source dataframe and well name')
        
        return None
    
    
    #Otherwise, use create_plot_with_df to create the well plot
    fig, ax, axs_dict = create_plot_with_df(well_df,
                                            well_name,
                                            min_depth,
                                            max_depth,
                                            save_fig,
                                            **kwargs)
    
    return (fig, ax, axs_dict)
    
def create_plots_from_dict(multi_well_df, conditions_dict, **kwargs):
    """
    Use a dictionary of conditions in the form {UWI: {param1: param1_value}}
    to make muliple plots in a multi-well dataframe. Useful for specifying
    depths for multiple wells
    (eg. min_depth, max_depth, save_fig) to

    Parameters
    ----------
    multi_well_df : DataFrame
        The dataframe with ASCII data from multiple wells (eg. mega_df).
    conditions_dict : dict
        A dictionary of conditions to apply to the multi well dataframe

    Returns
    -------
    figure_axes_dict : tuple
        A tuple with the figure, axes and axes dictionary objects passed

    """
    figure_axes_dict = {}
    
    for UWI, conditions_dict in conditions_dict.items():
        if not conditions_dict:
            fig, ax, axs_dict = create_plot_filter_name_fr_df(multi_well_df, UWI)
            
            figure_axes_dict[UWI] = (fig, ax, axs_dict)
        else:
            min_depth = conditions_dict.get('min_depth', None)
            max_depth = conditions_dict.get('max_depth', None)
            save_fig = conditions_dict.get('save_fig', False)
            
            fig, ax, axs_dict = create_plot_filter_name_fr_df(multi_well_df, UWI, min_depth=min_depth,
                                          max_depth=max_depth, save_fig=save_fig)
            figure_axes_dict[UWI] = (fig, ax, axs_dict)
    
    return figure_axes_dict
        
def replace_ascii_data_for_classification(well_log, ML_classification_df):
    
    well_log_data = well_log.A
    
    ML_classification_df = ML_classification_df.set_index(['Well_name', 'DEPT'])
    
    #Get the depths of the curves in the ML classification dataframe
    curve_depths_index = ML_classification_df.index.get_level_values(1) #Get only depths from multiindex    
    curve_depths_list = curve_depths_index.to_list()    
    
    #Define top and base
    top = min(curve_depths_list)
    print(f'Top is {top}')
    base = max(curve_depths_list)
    print(f'Base is {base}')
    
    curve_limits = (top, base)
    
    #Adjust curve limits attribute to new dataframe
    well_log.curve_limits = curve_limits
    
    well_log.A = ML_classification_df    
                                                       
    return
    
def get_ML_classification_data(well_name, well_log, ML_classification_url):
    
    ML_classification_df_all_wells = pd.read_csv(ML_classification_url)
    
    ML_classification_df_well = ML_classification_df_all_wells[ML_classification_df_all_wells['Well_name'] == well_name]
    
    return ML_classification_df_well
    
    
    



if __name__ == "__main__":
    
    #Define the name of the well to be created (Currently hard-coded)
    well_name = 'Kona-6'
    
    #Create path based on well name
    # well_path = r"C:\Programming\AI_Production_Log_Project_2023\Scripting Session Folders\08MAR2024 Bypassed Pay Session\Logs" + "\\" + well_name + ".las"
    
    well_path = r"C:\Programming\AI_Production_Log_Project_2023\Scripting Session Folders\08MAR2024 Bypassed Pay Session\Logs\AKIRA-8.las"
    
    #Use plastic module to create Log object
    well_log = merge.main_test()
    
    #TODO Fix error handling below
    # #Check for error (Not working)
    # if well_log == FileNotFoundError:
    #     print('File was not found')  
    #     raise FileNotFoundError()
    
    #TODO fix below so that the functions using this variable access the
    #attribute directly from the Log object
    
    #Assign the ascii data from the Log object to a variable
    well_log_A = well_log.A
        
    #Designate whether there is a machine learning output to override the
    #ascii data
    replace_log = False
    
    if replace_log == True:
        #Access ML output
        
        #ML output URL
        ML_classification_url = r'C:\Programming\AI_Production_Log_Project_2023\Scripting Session Folders\08MAR2024 Bypassed Pay Session\ML_classification_log_data_15MAR2024.csv'
        
        #Load ML output to dataframe
        ML_classification_df = get_ML_classification_data(well_name, well_log, ML_classification_url)
        
        #Replace ascii data in log object with ML dataframe
        replace_ascii_data_for_classification(well_log, ML_classification_df)
        
    #TODO fix below so that the functions using this variable access the
    #attribute directly from the Log object
    
    #Explicitly assign the curve limits to well_log_limits variable
    well_log_limits = well_log.curve_limits
    
    #Create the visual plot    
    fig, ax, axs_dict = create_plot(well_log, well_name)

    #Set current axes to make sure the label in the next step is drawn on the
    #correct figure
    plt.sca(ax)
    
    #Add well name at bottom of plot
    fig.text(0, 0, well_name, fontsize=16)
    

#%%

    #plt.gca().invert_yaxis()
    
