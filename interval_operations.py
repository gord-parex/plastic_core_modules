# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:05:12 2024

@author: gfoo
"""

import pandas as pd
import numpy as np
from . import plastic as pl
from . import meta_plastic as mpl



def parse_interval(well_interval):
    top, bottom = well_interval.str.split()


def get_interval_df_data(df, well_name, perf_interval):
    top, base = perf_interval
    
    try:
        slice_data = df.loc[(well_name, top):(well_name, base)]
    except:
        print('Error slicing data')
        return None
    
    return slice_data


def extract_interval_data_entry(mega_df, well_interval_info):

    """
    ('LMAM0001', '12100-12125*12150-12160')
    """

    well_name, interval_string = well_interval_info

  
    #Take raw string and split on asterisk into list of strings
    perf_list = str.split(interval_string,'*')
    #Take each string perf interval and split on hyphen into a nested list
    perf_list = [str.split(i,'-') for i in perf_list]
    #For each value in nested list, convert to float
    parsed_perf_interval_list = [list(map(float, map(lambda x: x.strip(), i))) for i in perf_list]

    interval_data_list = []    

    for perf_interval in parsed_perf_interval_list:
        interval_data = get_interval_df_data(mega_df, well_name, perf_interval)
        # if not interval_data: #Check if interval data returns None
        #     continue
            
        interval_data_list.append(interval_data)

    concat_df = pd.concat(interval_data_list)
        

    return concat_df


def extract_interval_data_entries(mega_df, well_interval_info_list):
    """
    [('LMAM0001', '12100-12125*12150-12160'), ('LMAM0002', '13070-13115*14650-14670')]
    """
    
    df_list = []
    
    for interval_entry in well_interval_info_list:
        entry_df = extract_interval_data_entry(mega_df, interval_entry)
        df_list.append(entry_df)
    
    concat_df = pd.concat(df_list)
    
    return concat_df

    
def flag_large_df_with_smaller_df(large_df, small_df, flag_name='Flag'):
    """Create a flag column in larger dataframe based on the multiindex values
    of a smaller dataframe. This is useful to create a flag in the larger
    dataframe for all the rows that are present in the smaller dataframe."""
    
    # Create a set of tuples from the multiindex of the small dataframe
    index_set = set(small_df.index)
    
    # Initialize the flag column with 0 (indicating 'not flagged')
    large_df[flag_name] = 0
   
    # Apply the flag where the index of large_df is found in the index_set
    # This can be done efficiently using the .loc accessor and the .index.isin() method
    large_df.loc[large_df.index.isin(index_set), flag_name] = 1

    return large_df
   