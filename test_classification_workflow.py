# -*- coding: utf-8 -*-

#Plastic --> Metaplastic -->ASCII data helper -->test_classification_extraction_workflow


"""
Created on Wed Feb 22 10:41:19 2023
@author: Gordon Foo 

THE 22FEB2024 version is STABLE

UPDATED 15FEB2024
-Included functionality to replace values with nans (will help with statistical
 analysis so that null values do not impact averages etc.) for future
 manipulation
 -Eliminated step that replaces nans with LAS values for nulls (commented out)
 
 
Redundant or unnecessary methods:
-Concat results

"""




""" 
Test classification extraction workflow

This file takes an excel file with a summary of test results and parses it a
number of different ways. The end result is a list of tuples, where each entry
corresponds to a test record in the original 'test_classification' excel 


parse_perf_intervals() collects the information on
perf intervals from the test_classification spreadsheet and leaves it in a form
that can be used to slice out log data.

The primary function of extract_log_data() is to take the parsed perf intervals
from each test and slice out the corresponding data in the logs. It also helps
to store the metadata from the test (well, formation, etc.) as a tuple which
includes the sliced-out log data.


"""
import pandas as pd
import pdb
import numpy as np

from . import plastic as pl

#test_classification = pd.read_excel(r"C:\Users\gfoo\Documents\plastic-master\plastic-master\16FEB2023 Bypassed Pay Session\WELL_CLASSIFICATION_IA_11OCT2022.xlsx")

#test_classification = test_classification.set_index(['WELL','FORMATION'])


def parse_perf_intervals(raw_perf_string):
    """Parse Perf Intervals
    This function takes a raw perforation string from the test classification
    summary spreadsheet and splits it into a list. First, it is split on the
    asterisk (*) character to get the unique shot clusters, then it splits on
    the hyphen character (-) to split the shot cluster into a 2 separate
    values. The return is a list of lists, with each sublist featuring a shot
    cluster for that particular test.
    """
    #Take raw string and split on asterisk into list of strings
    perf_list = str.split(raw_perf_string,'*')    
    #Take each string perf interval and split on hyphen into a nested list
    perf_list = [str.split(i,'-') for i in perf_list]
    #For each value in nested list, convert to int
    parsed_list = [list(map(float, map(lambda x: x.strip(), i))) for i in perf_list]

    return parsed_list
    
#test_classification['Parsed Perf Interval'] = test_classification['INTERVAL'].apply(parse_perf_intervals)

def extract_log_data(test_classification_df, master_df, parsed_perf_column='Parsed Perf Interval'):
    """Extract Log Data
    Extract log data takes a test classification dataframe and extracts the
    log data associated with all of the perf intervals in the indivitual tests.
    The end product is a list of lists, where each sublist contains a well
    name, the formation the test was performed in, the classification of the
    test (0 for economic failure, 1 for economic success), and a list with the
    ASCII log data from the perforated interval(s) in the test.
    
    Args:
        test_classification_df(dataframe): A dataframe with the original
            production test data.
        master_df(dataframe): A dataframe with all the required ASCII log data
            from the original set of logs corresponding to the test
            classification dataframe
        parsed_perf_column(str): The label for the column in the test
            classification dataframe where the raw perf interval string can
            be found
        
    """
    #Instantiate list of lists
    list1 = []
    
    row_counter = 0
    test_classification_well_names = get_unique_index_values(test_classification_df, 'UWI') #Find the unique list of well names in test_classification_df
    master_df_well_names = get_unique_index_values(master_df, 'Well_name') #Find the unique list of well names in master_df
    common_names = list_intersection(test_classification_well_names, master_df_well_names) #Compare test_classification_df and master_df well names, assign unique values
    print('The common names are ', common_names)
    for index, row in test_classification_df.iterrows(): #Iterate over rows in test_classification_df
        well_name = index[0] #Position of well name in index
        if well_name in common_names:
            #print('Well Name in extract log data ', well_name)
            perf_interval = row[parsed_perf_column]
            print('Perf interval in extract log data is ', perf_interval)
            formation = index[1] #Position of formation name in index
            sequence = row['SEQUENCE']
            classification = row['CLASSIFICATION']
            filt_well_data_df = master_df.loc[master_df.index.get_level_values('Well_name') == well_name]
            #print(filt_well_data_df)
            perf_interval_data = well_slice_perf_interval_data(filt_well_data_df, well_name, perf_interval)
            perf_interval_data['Test Number'] = row_counter + 1 # Add column with test number in it, which is row counter
            perf_interval_data['Sequence'] = sequence
            perf_interval_data['Formation'] = formation #Add column with formation name
            list1.append((well_name, formation, classification, (row_counter +1), perf_interval_data)) #Row counter is a reference to which test the line is in the table
            row_counter += 1
            #print(row_counter)
    
    return list1

def get_unique_index_values(dataframe, index_name): #Used in extract_log_data()
    """ Get unique index values
    Get unique index values for a dataframe
    
    dataframe is the dataframe that you are passing to extract the index names from
    index_name is the name of the column in the multi index that you are
    trying to extract the unique values from
    """
    unique_values = dataframe.index.get_level_values(index_name)
    
    return unique_values


def list_intersection(list1, list2): #Used in extract_log_data()
    """List intersection
    Return the values that 2 lists have in common
    """
    common_values = set(list1).intersection(list2)
    
    return list(common_values)


def well_slice_perf_interval_data(filt_well_data_df, well_name, perf_intervals): #Used in extract_log_data()
    print('Well name in slice perf interval data', well_name)
    print('Perf intervals in slice perf interval data', perf_intervals)
    print('Length of perf intervals in slice perf interval data', len(perf_intervals))    
    #print(perf_intervals[0][0])
    well_df = filt_well_data_df.loc[(well_name, perf_intervals[0][0]):(well_name, perf_intervals[0][1])] #Try to index dataframe with perf interval values.
    if len(perf_intervals) > 1:
        print('Going to multiple perfs')
        for i in range(len(perf_intervals))[1:]:
            temp_df = filt_well_data_df.loc[(well_name, perf_intervals[i][0]):(well_name, perf_intervals[i][1])]
            well_df = pd.concat([well_df, temp_df], axis=0)
            print('Concatenated frame')
        
    print('Slice is ', well_df)
    #well_df = replace_nan_with_value(well_df) #Replace NaN values with -999.25 null values.
    return well_df



def replace_nan_with_value(dataframe, null_value=-999.25): #Used in well_slice_perf_interval_data()
    """
    Replace nan values in a dataframe with another value, as the nans do not
    allow the dataframes to be concatenated.
    Args:
        dataframe is the dataframe where you would like to replace NaNs.
        null_value is the value to replace the NaNs The default value for replacing NaNs is -999.25
    """
    dataframe_no_nan = dataframe.fillna(null_value) #Replace nans with null value specified
    
    return dataframe_no_nan

def replace_nan_extracted_log_data(extracted_log_data_list):
    for i,j,k,l,m in extracted_log_data_list:    
        m = m.replace(-999.25, np.nan)
        
    return extracted_log_data_list
    
    
def forward_fill_log_data_list(extracted_log_data_list, depth_step=0.5):
    """Take a log data list (this is the product of extract log data) and
    apply the forward fill and pruning routine to the dataframe with sliced
    out data"""
    modified_list = [(i,j,k,l, pl.forward_fill_and_resample_pruning(m, depth_step))
                      for i,j,k,l,m in extracted_log_data_list]    
    
    return modified_list



def concat_results(extracted_log_data_list):
    """
    Take the extracted data list and concatenate all results into one
    table. Option to perform forward fill on each dataframe
    """
    concat_df = pd.DataFrame()
    for well_name, corex_strat_unit, test_result, test_number, df in extracted_log_data_list:    
        df['Test_result'] = test_result
        df['Test_number'] = test_number
        concat_df = pd.concat([concat_df, df])
    
    return concat_df




#### Quarantined version of well_slice_perf_interval_data ######
"""
def well_slice_perf_interval_data(filt_well_data_df, well_name, perf_intervals):
    fail_count = 0
    success_count = 0
    try:
        well_df = filt_well_data_df.loc[(well_name, perf_intervals[0][0][0]):(well_name, perf_intervals[0][0][1])] #Try to index dataframe with perf interval values.
        print('Initial well DF is ', well_df)
        for i in range(len(perf_intervals[0]))[1:]:
            temp_df = filt_well_data_df.loc[float(perf_intervals[0][i][0]):float(perf_intervals[0][i][1])]
            well_df = pd.concat([well_df, temp_df], axis=0)
            success_count += 1
            print('Success count ', success_count)
            
    except Exception as e:
        print('Failed. First perf inteerval is', perf_intervals[0][i][0], 'and second perf interval is ', str(perf_intervals[0][i][1]))
        fail_count += 1
        print('Fail count ', fail_count)
    return well_df
"""  
    
    # if len perf_intervals <= 1:
    #     filt_well_data_df_sliced = filt_well_data_df.loc[float(perf_intervals[0][0]):float(perf_intervals[0][1])]
    # else:
    
    
    
    
    # well_df = well.A.loc[float(perf_intervals[0][0]):float(perf_intervals[0][1])]
    # if len(perf_intervals) > 1:
    #     for i in range(len(perf_intervals))[1:]:
    #         temp_df = well.A.loc[float(perf_intervals[i][0]):float(perf_intervals[i][1])]
    #         well_df = pd.concat([well_df, temp_df], axis=0)
    
    # return well_df    
"""
import pickle

with open('log_data.pk1', 'wb') as f:
    pickle.dump(extracted_log_data, f)
    
with open('log_data.pk1', 'rb') as f:
    extracted_log_data = pickle.load(f)
    
"""