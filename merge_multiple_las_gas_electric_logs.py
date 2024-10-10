# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:45:09 2024

@author: Gfoo
"""

#Imports

import os
import numpy as np
import pandas as pd
import datetime

from . import plastic as pl
from . import meta_plastic

def get_raw_file_list(logs_dir):
    """
    Use os.walk to generate a list of .las filenames in the logs directory

    Parameters
    ----------
    logs_dir : str
        Log directory URL.

    Returns
    -------
    raw_file_list : List
        A list of all the .las files located in the logs directory.

    """
    
    raw_file_list = []
    
    for root, dirs, files in os.walk(logs_dir):
        for file in files:
            raw_file_list.append(file)
        
    
    #Use list comprehension to ensure that only .las files enter the list
    raw_file_list = [file for file in raw_file_list if
                     str(file).endswith('.las')]
        
    return raw_file_list


def get_cleaned_list_of_log_files(raw_file_list):
    """
    Take a list of raw filenames generated in os.walk(), and determine 
    the unique UWIs that exist within this list

    Parameters
    ----------
    raw_file_list : list
        The list of raw .las filenames gathered by os.walk.

    Returns
    -------
    split_list_unique : list
        The list of unique UWIs found in the 'raw_file_list' list.

    """
    #Split on .las
    split_list_no_las = [i.split('.las')[0] for i in raw_file_list]
    
    #Split on underscore
    split_list_no_underscore = [i.split('_1')[0] for i in split_list_no_las]
    
    #Set operation to remove duplicates
    split_list_set = set(split_list_no_underscore)
    
    #Convert set back to list
    split_list_unique = list(split_list_set)
    
    #Sort alphabetically
    split_list_unique.sort()
    
    return split_list_unique


def get_associated_logs_dict(unique_log_name_list, raw_file_list):
    
    """Get a dictionary that groups all of the ossociated logs together based
    on which .las files in the directory contain the text string of the base 
    log name

    Parameters
    ----------
    unique_log_name_list : list
        The list of unique log names to reference.
    raw_file_list : list
        The list of raw .las filenames found in the specified logs folder.

    Returns
    -------
    logs_name_dict : dict
        A dictionary that relates the UWI (key) to the list of .las filenames
        that are associated with that UWI (value).

    """
    
    
    #Instantiate dictionary to store results
    logs_name_dict = {}
    
    #Loop over the unique logs (base logs) in the unique_log_name_list
    for entry in unique_log_name_list:
        
        
        #Instantiate list that will be associated with each key
        logs_name_dict[entry] = []
        

        for file in raw_file_list:
            
            #Split the filename on underscore and take the first part of
            #the filename
            file_split_underscore = file.split('_')[0]
            file_split_underscore_period = file_split_underscore.split('.')[0]
            
            #If this first part of the split filename is equal to the
            #entry, add the filename to the list
            if (file_split_underscore_period == str(entry)):
                  logs_name_dict[entry].append(file) 

        
    return logs_name_dict
    
def delete_df_columns(list_to_delete, dataframe):
    """Delete columns based on a list of columns to delete. Accepts a list of
    columns to delete and a dataframe and returns a dataframe"""
    for i in list_to_delete:
        dataframe.drop(i, axis=1, inplace=True)
        
    return dataframe


def create_log_object_from_files(well_key, las_list_for_log, logs_dir,
                                 ffill_only_gas_curves_oil_show = False):
    """
    This function accepts a UWI (well_key) and a list of las files associated
    with that well key. A Log object is created initially. Then, if there is
    more than one .las file in the list, the data in the .las filesis merged
    through a join, and this merged data becomes the ASCII data in the Log
    object
    

    Parameters
    ----------
    well_key : str
        Generally the UWI of the well for which the Log object is created.
    las_list_for_log : list
        A list of .las filenames in the log directory that are associated with
        the well key.
    logs_dir : str
        The directory where log files are stored
    ffill_gas_curves_oil_show : TYPE, optional
        This boolean parameter specifies whether to perform forward fill on
        the curves in the ASCII data (True) or not (False). The default is True.

    Returns
    -------
    main_log : Log object
        This is the log object corresponding to the well_key (UWI) with the
        fully merged data.

    """
    gas_log_mnems = ['C1_GAS', 'C2_GAS', 'C3_GAS', 'IC4_GAS', 'IC5_GAS',
                     'NC5_GAS', 'TOTALGAS', 'NC4_GAS', 'OIL_SHOW']    
    
    #Determine how many logs are associated with the base log. Len = 1
    #means that only the base log is associated with the base log
    log_list_length = len(las_list_for_log)
    
    
    #If len = 1, in other words if the only log is the base log, create
    #a log object for the base log
    if log_list_length == 1:
                   
        log_fn = logs_dir + '\\' + well_key + '.las'
        
        #Store the log object with the base log as the key
        main_log = pl.create_log(log_fn, well_key)
        
    #If there is more data than just the base log...
    elif log_list_length > 1:
        
        #The name for the base log is the first entry in the log list
        #assicuated with the key
        base_name = las_list_for_log[0]
        
        #Create the log object for the base log
        main_log_fn = logs_dir + '\\' + well_key + '.las'
        main_log = pl.create_log(main_log_fn, well_key)            
        
        #Assign the ASCII data to a variable and prepare to join to other
        #logs
        
        log_data = main_log.A
        
        
        #In preparation to loop over the other logs associated with the
        #base log, we don't need the first entry any more. Take a slice of
        #the log list that doesn't include the first entry
        shortened_log_list = las_list_for_log[1:]
        
        
        #Loop over the rest of the logs starting with the 2nd entry
        for index, other_log_name in enumerate(shortened_log_list):
            
            #Create a log object for the i-ith entry
            other_log_fn = logs_dir + '\\' + other_log_name                
            other_log = pl.create_log(other_log_fn, well_key)
            
            
            #Assign the ASCII data from the i-ith log object to a variable
            other_log_data = other_log.A
            
            
            #Use an "outer join" to join the data of the i-ith log to the
            #base log
            log_data = log_data.join(other_log_data, 
                          how='outer', 
                          lsuffix=f'_{index}_left',
                          rsuffix=f'_{index}_right'
                          )
        
        row_number_columns_to_delete =  [i for i in log_data.columns if 'Row_number' in i]
        
        log_data = delete_df_columns(row_number_columns_to_delete, log_data)
        
        #Determine whether all curves will be forward filled, or just gas
        #curves
        if ffill_only_gas_curves_oil_show == True:
        
            ffill_mnems = gas_log_mnems
        
        #If the ffill_only_gas_curves_oil_show parameter is false, get all
        #column names as a list, and forward fill all columns
        else:
            ffill_mnems = log_data.columns.to_list()
            
            # for curve in gas_log_mnems:
            #     try:
            #        log_data.loc[:,curve].ffill(axis=1, inplace=True)
            #     except KeyError:
            #         continue
                
        
        for curve in ffill_mnems:
            try:
                log_data.loc[:,curve].ffill(inplace=True, limit=4)
            except KeyError:
                continue
        
            
        
        #Once all of the log data has been joined, assign this joined log
        #data to the "A" attribute of the base log
        main_log.A = log_data
        
    return main_log
    
    

def assign_multiple_log_objects_to_dict(logs_name_dict, logs_dir, ffill_gas_curves_oil_show=True):
    """
     This function takes a dictionary of UWI names and their associated
     .las filenames, and converts it into a dictionary of log objects with 
     merged curve data from the associated .las files.

    Parameters
    ----------
    logs_name_dict : dict
        A logs name dict is a dictionary that stores a UWI as a key, and the 
        value is a list of .las files in the specified logs directory that are 
        associated with that UWI. 
        
        E.g.: {'KONA0006': ['KONA0006.las', 'KONA0006_1.las']}
        
    logs_dir : str
        The URL location of the log files
    
    ffill_gas_curves_oil_show : TYPE, optional
        This boolean parameter specifies whether to perform forward fill on
        the curves in the ASCII data (True) or not (False). The default is True.

    Returns
    -------
    log_object_dict : dict
        Returns a dictionary with the UWI as the key, and a log object as the
        value.

    """
    
    #A true value for ffill_gas_curves_oil_show will forward fill the gas curve
    #and oil show curve values after all dataframes have been joined
   
    
    #Instantiate dictionary that will hold log objects to be merged
    log_object_dict = {}
    
    #Loop over keys (base logs) in logs_name_dict{}
    for key in logs_name_dict.keys(): 
        
        #log_list_for_key[] is the list of logs associated with the base log
        log_list_for_key = logs_name_dict[key]
        
        main_log = create_log_object_from_files(key, log_list_for_key, logs_dir)
        
        #Assign this new log object to the log object dictionary
        log_object_dict[key] = main_log
        
        print(key)

    #Return the log object dictionary            
    return log_object_dict
            
def audit_curves(log_objects_dict):
    """
    Uses groupby objects and the groupby.mean() methods to show which logs have
    missing curve data. No value in the CSV file shows that the curve does not
    exist in the given log

    Parameters
    ----------
    log_objects_dict : dict
        A dictionary with a "UWI: Log object" key-value pair.

    Returns
    -------
    No return, but a CSV file in the current working directory is written in
    the format 'Joined_logs_mean {today's date}.csv'

    """
    
    today = datetime.date.today()
    
    filename = f'Joined_logs_mean {today}.csv'
    
    super_df = meta_plastic.create_super_df(log_objects_dict) 

    gb1 = super_df.groupby('Well_name')

    gb1_mean = gb1.mean()

    gb1_mean.to_csv(filename)
    

def resample_log_object(log_object, step_value=0.5):
    """Resample the curve data in a log object with a given depth step. 
    Default is 0.5"""
    log_ascii_data = log_object.A
    
    log_ascii_data = pl.resample_depth_step_pruning(log_ascii_data, step_value)
    
    log_object.A = log_ascii_data
    

def resample_log_dictionary(log_dict, step_value=0.5):
    """Resample the curve data in log objects in a log dictionary with a given 
    depth step. Default is 0.5"""
    for well, log_object in log_dict.items():
        log_object = resample_log_object(log_object, step_value)
            
def ffill_resample_prune_log_dictionary(log_dict, step_value=0.5):
    """Forward fill and resample the curve data in log objects in a log 
    dictionary with a given depth step. Default is 0.5"""
    for well, log_object in log_dict.items():
        ascii_data = log_object.A
        ascii_data = pl.forward_fill_and_resample_pruning(ascii_data)

    
def create_well_log_dict_with_merged_logs(logs_dir):
    #Get raw file names with os.walk()
    raw_file_list = get_raw_file_list(logs_dir)
    
    #Clean up list of files
    cleaned_list_of_log_files = get_cleaned_list_of_log_files(raw_file_list)
    
    #Get the dict of the filenames ossiciated wit the base log
    associated_logs_dict = get_associated_logs_dict(cleaned_list_of_log_files,
                                                    raw_file_list)
    
    #Get the dictionary of log objects
    log_objects_dict = assign_multiple_log_objects_to_dict(associated_logs_dict,
                                                            logs_dir)
    
    resample_log_dictionary(log_objects_dict)
    
    return log_objects_dict


def main(logs_dir=None, working_dir=None):
    
    # #Ensure these point to the local directory
    # logs_dir = os.getcwd() + r'\Logs' #Logs directory is subfolder of current working directory. Concatenate these strings to create path
    
    
    if logs_dir == None:
        logs_dir = r'C:\Programming\AI_Production_Log_Project_2023\Scripting Session Folders\15APR2024 Bypassed Pay Session\Logs' # #Logs directory is subfolder of current working directory. Concatenate these strings to create path
    
    #Assign correct working directory, precise location will depend on where directory is unzipped
    if working_dir == None:
        working_dir = r'C:\Programming\AI_Production_Log_Project_2023\Scripting Session Folders\15APR2024 Bypassed Pay Session'
    
    os.chdir(working_dir) #Change working directory
    
    #Get raw file names with os.walk()
    raw_file_list = get_raw_file_list(logs_dir)
    
    #Clean up list of files
    cleaned_list_of_log_files = get_cleaned_list_of_log_files(raw_file_list)
    
    #Get the dict of the filenames ossiciated wit the base log
    associated_logs_dict = get_associated_logs_dict(cleaned_list_of_log_files,
                                                    raw_file_list)
    
    #Get the dictionary of log objects
    log_objects_dict = assign_multiple_log_objects_to_dict(associated_logs_dict,
                                                            logs_dir)
    
    resample_log_dictionary(log_objects_dict)
    
    return log_objects_dict
    
#     Kona_6 = log_objects_dict['KONA0006']
    
#     return Kona_6
    
# if __name__ == "__main__":
#     kona_6 = main_test()
    
   

    