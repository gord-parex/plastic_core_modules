# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:49:36 2024

@author: Gfoo
"""

import os
import pandas as pd
import numpy as np

def find_well_in_df(well_info_df, well_name_to_find):
    """
    Locate where a particular well name is (in string format) in a dataframe
    
    Parameters
    ----------
    well_info_df : DataFrame
        The dataframe containing the consolidated well information, including
        the variety of names for each well that will be searched.
    string_to_find : string
        The string to search for in the dataframe

    Returns
    -------
    The index and column of the located cell, if the string is found.
    Otherwise, no return

    """
    
    #Loop over columns
    for column in well_info_df.columns:
        
        #Loop over items in columns
        for index, value in well_info_df[column].items():
            
            #If value found, return the index (row) and the column it was found in
            if value == well_name_to_find:                
                return (index, column)  # Return the row index where a match is found
            
            #Otherwise, return a tuple with None
            else:                
                continue

def find_well_in_series(well_name_series, well_name_to_find):
    """Get the row index where a well name (in string format) is found in a
    series
    """
    
    print(f'Well name to find is {well_name_to_find}')
    #Loop over items in series
    for index, well_name in well_name_series.items():
        
        #If value found, return the index (row) it was found in
        if well_name == well_name_to_find:
           
            return index  # Return the row index where a match is found
        
        else:
            continue
        
    print('Nothing found')    
    return None
        
def get_primary_key(df, index, primary_key_column_name='UWI'):
    """
    Retrieve the primary key (UWI) from a dataframe
    """
    primary_key = df.loc[index][primary_key_column_name]
    
    return primary_key

def get_anh_well_name_key(df, index):
    anh_well_name_key = df.iloc[index]['WELL_NAME']
    
    return anh_well_name_key
    
def test_well_name_to_UWI(input_df, well_info_df,
                          test_well_name_column='WELL',
                          uwi_column_name='UWI'):
                          
    """
    Take a dataframe with Corex well names and add a column to tie it back
    to the universal UWI through the consolidated well info spreadsheet

    Parameters
    ----------
    input_df : DataFrame
        The dataframe with the table that needs a universal UWI column added
    corex_name_column : (Optional) String
        The column header for the Corex well name to use

    Returns
    -------
    df : DataFrame
        The original dataframe with a column for universal UWI, if the entries
        could be found from the original input data

    """
    
    #Instantiate error log
    error_log = []
    
    #Add empty column to populate UWI
    input_df[uwi_column_name] = ""
    
       
    #Specify the well name column for the consolidated well info dataframe
    well_info_test_name_column = 'COREX_WELL_NAME'
    
    #Slice out the names of the wells in the test classification dataframe
    well_names_for_test_classification = input_df.loc[:,test_well_name_column]
    
    #Slice out the names of the wells in the consolidated well info file
    well_names_for_consolidated_info = well_info_df.loc[:, well_info_test_name_column]
        
       

    #Loop over the series with well names from the test classification sheet
    for test_info_index, test_info_well_name in well_names_for_test_classification.items():
        
        #Find the well name (from the test classification file) in the
        #consolidated well information series
        index = find_well_in_series(well_names_for_consolidated_info,
                                        test_info_well_name)
        
        #Check if None was returned. If the value is something other than 
        #None, get the primary key associated with this index number (in this
        #case the primary key is 'UWI')
        if index != None:
            UWI = get_primary_key(well_info_df, index)
            print(f'UWI is {UWI}')
            
            #Assign the UWI string to the cell that corresponds to the UWI for
            #the test in the test classification dataframe
            input_df.at[test_info_index, uwi_column_name] = UWI
        
        #If there is no match, add information on the test to error log
        else:
            error_log.append(str(input_df.loc[test_info_index]))
        
            continue
    
    #Return modified test classification dataframe and the error log
    return (input_df, error_log)
    

if __name__ == "__main__":
    
    #Assign correct working directory
    os.chdir(r'C:\Programming\Other Scripts\Well Name Conversion') #Change working directory
    
    #Consolidated data filename
    well_info_df_filename = "Consolidated_Well_Info 05APR2024.xlsx"
    
    #Load well info dataframe
    well_info_df = pd.read_excel(well_info_df_filename)
    
    
    input_df1 = pd.read_csv('26FEB2024_WELL_CLASSIFICATION_IA_TIGANA_CSV.csv')
    
    output_df = test_well_name_to_UWI(input_df1, well_info_df)


#TODO: Finish function below
#def UWI_to_well_name

# if __name__ == "__main__":

#     index, column = find_well_in_df(well_name_original_df, 'ABAN0004')
    
#     primary_key = get_primary_key(well_name_original_df, index)
                        
#     anh_well_name_key = get_anh_well_name_key(well_name_original_df, index)
    
    
#     print(f'Primary key is {primary_key}')
#     print(f'ANH Well Name Key is {anh_well_name_key}')




