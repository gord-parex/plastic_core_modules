# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:46:16 2024

@author: Gfoo
"""

import pandas as pd
import numpy as np
import pickle as pkl

from . import plastic as pl

def get_unique_wells(well_column):
    """Take a well name column from a dataframe and return a list with the
    unique well names"""
    
    unique_well_list = list(well_column.unique())
    unique_well_list.sort()
    
    return unique_well_list

def get_unique_formation_tops(formation_top_column):
    unique_formation_top_list = formation_top_column.unique().to_list()
    
    return unique_formation_top_list


def get_permitted_tops(permitted_tops_and_bases_filename):
    permitted_tops_and_permitted_base = pd.read_csv(permitted_tops_and_bases_filename)

    permitted_tops = permitted_tops_and_permitted_base['UNIT_NAME']
    
    permitted_tops = (permitted_tops.squeeze()).to_list()
    
    return permitted_tops

def top_interpreter_pick(tops_df, permitted_tops_and_bases_filename, 
                         interpreter_preference_list, uwi_column='UWI',
                         depth_column='MD'):
    """Takes a dataframe with information on formation tops, a list of
    formation tops that should be searched for and a list of interpreters in
    order of preference from highest to lowest and pulls out the a formation\
    top for each of the specified tops from the list which corresponds to the
    most-preferred interpreter available for that well-formation top combination
    

    Parameters
    ----------
    tops_df : DataFrame
        DataFrame with all aggregated formation top information.
    formation_top_list : List
        List of the formation tops that should be serarched for in the aggregated
        formation top dataframe
    interpreter_preference_list : List
        A list with the interpreters that should be attempted in the search,
        ordered from highest-preference to lowest preference
    uwi_column : str, optional
        The name of the column correspondeing to UWI in the tops dataframe

    Returns
    -------
    filtered_top_list : List
        A list of rows that were successfully retrieved from the search

    """
    
    #Get the list of permitted tops from the permitted tops and bases filename
    permitted_tops = get_permitted_tops(permitted_tops_and_bases_filename)
    
    #Specify depth column
    depth_column = 'MD'
    
    #Drop rows that have empty depth values
    tops_df = tops_df.dropna(subset=[depth_column])
    
    #Extract the column that corresponds to the UWI
    wells_column = tops_df[uwi_column]
    
    #Run get unique wells to get a list of the unique UWIs in the column
    unique_well_list = get_unique_wells(wells_column)
    
    #Instantiate a list to store filtered tops    
    filtered_top_list = []
    
    #Loop over wells in DF
    for well in unique_well_list:
        
        #Filter out rows that belong to the particular well being looped over
        well_df = tops_df[tops_df[uwi_column] == well]
        
        #Loop over the permitted tops
        for formation in permitted_tops:
            #Try to index out rows that belong to the particular, permitted
            #well tops name
            formation_well_df = well_df[well_df['Pick Name'] == formation]
            
            #If there are no rows returned, continue to the next formation in 
            #the loop. Otherwise, continue to execute code below            
            if formation_well_df.size == 0:
                continue
            
            #Get length of permitted interpreter list
            interpreter_len = len(interpreter_preference_list)
            
            #Loop over interpreter preference list
            for interpreter in interpreter_preference_list:
                #In this well-specific dataframe, try to index values with
                #the particular interpreter included
                formation_well_interpreter_record = formation_well_df[formation_well_df['Author'] == interpreter]
                
                #If nothing comes up, continue to next interpreter
                if formation_well_interpreter_record.size == 0:
                    continue
                
                #Otherwise, append the row with the preferred interpreter to
                #the filtered top list
                else:
                    filtered_top_list.append(formation_well_interpreter_record)
                    break
    
    #TODO Check if the row below is necessary
    ordered_filtered_top_list = [df.sort_values(depth_column) for df in filtered_top_list]
    
    #Return the dataframe that 
    ordered_filtered_top_df = pd.concat(filtered_top_list)
            
    return ordered_filtered_top_df


#TODO This function in progress
def get_formation_tops_with_base(filtered_top_list, permitted_bases_for_tops):
    
       
    return # {well_1: {formation_1: (top_depth, base_depth), formation_2: (top_depth, base_depth)}, well_2: {formation_1: (top_depth, base_depth), formation_2: (top_depth, base_depth)}}


def load_permitted_tops_and_bases(permitted_tops_and_bases_fn,
                                  strat_unit_column_name='UNIT_NAME',
                                  permitted_base_column='PERMITTED_TOPS_BELOW'
                                  ):
    
    permitted_tops_and_bases = pd.read_csv(permitted_tops_and_bases_fn)
    
    split = lambda x: x.split(',') if isinstance(x, str) else []
    
    permitted_tops_and_bases[permitted_base_column] = permitted_tops_and_bases[permitted_base_column].apply(split)
    
    permitted_tops = permitted_tops_and_bases[strat_unit_column_name].to_list()
    
    permitted_bases = permitted_tops_and_bases[permitted_base_column].to_list()
    
    permitted_tops_and_bases_dict = dict(zip(permitted_tops, permitted_bases))
    
    return permitted_tops_and_bases_dict

    
 
def add_strat_unit_column(well_object, permitted_top_success_list, 
                          well_name_index='Well_name',
                          depth_index='DEPT', 
                          strat_unit_name='Strat_unit_name'):
    
    well_data_df = well_object.A
    
    if strat_unit_name not in well_data_df.columns:
        well_data_df[strat_unit_name] = np.nan
    
    
    for entry in permitted_top_success_list:
        success_top_name = entry[0]
        raw_top = entry[2]
        raw_base = entry [3]
        
        
        #Put the strat unit top and base through Plastic's find closest depth
        #method to ensure that there will be valid indicies for the slice
        verified_top, verified_top_index = pl.find_closest_depth_value(well_data_df, raw_top)
        verified_base, verified_base_index = pl.find_closest_depth_value(well_data_df, raw_base)
        
 
        #Used iloc here with the row number index because indexing with float
        #numbers is problematic. First, get the integer position of the strat
        #unit column
        
        strat_unit_name_integer_location = well_data_df.columns.get_loc(strat_unit_name)
        
        well_data_df.iloc[verified_top_index:verified_base_index,
                          strat_unit_name_integer_location] = success_top_name 
        
        #####well_data_df.set_index(['Well_name', 'DEPT'])
        
    setattr(well_object, 'A', well_data_df)
    
    return well_data_df
        
        
        
        
    
    
    
def permitted_top_checker(well_tops_df,
                          permitted_tops_dict,
                          strat_unit_column='Pick Name'):
    """Check a well tops dataframe to see if the strat unit names for the tops 
    belong to the permitted list of strat unit names. 
    
    Then, for each permitted top, check to see if there is a corresponding
    permitted base unit in the same well.
    
    If a permitted base is found, add a record to a list of results including
    the name of the top, the name of the corresponding base, and the top and
    base depths over which the strat unit name should apply"""
    
    #TODO Will have to do some handling to see whether to propogate last
    #formation, or not
    all_well_df_strat_unit_names = well_tops_df[strat_unit_column].to_list()
    
    #Instantiate success and failure lists
    permitted_top_success_list = []
    permitted_top_failure_list = []
    
    #Loop over rows in well tops dataframe
    for idx, row in well_tops_df.iterrows():
        
        strat_unit_name = row[strat_unit_column]
        print(strat_unit_name)
        
        #Check if the strat unit name is a permitted top. A permitted top
        #will be a key in the permitted tops dictionary
        if strat_unit_name in permitted_tops_dict.keys():
            permitted_bases = permitted_tops_dict[strat_unit_name]
            
            #If the top is allowed, loop over bases in order from shallowest to deepest    
            for permitted_base in permitted_bases:
                
                #IF the permitted base is in the list of all strat unit names
                #for the well, execute code below
                
                if permitted_base in all_well_df_strat_unit_names:
                    #The top depth for the interval is the MD value of the row
                    top_depth = row['MD']
                    
                    #Fancy index on the rows where the permitted base is in the
                    #well tops dataframe
                    row_for_base = well_tops_df[well_tops_df[strat_unit_column] == permitted_base]
                    
                    #Even though the Series should only have 1 row, the
                    #indexing is necessary here to pull the value out
                    base_depth = row_for_base.iloc[0]['MD']
                    
                    #Create tuple of data to append to the success list
                    permitted_top_result_data = (strat_unit_name,
                                                 permitted_base,
                                                 top_depth, base_depth)
                    
                    #Append to success list and break out of loop to look at
                    #next top in the well_tops_df
                    permitted_top_success_list.append(permitted_top_result_data)
                    break
                #If the permitted_base is not in the list of tops, continue to
                #the next permitted base
                else:                    
                    continue
            #If the permitted bases loop completes without a break, no valid
            #base was found. Append this to the failure list
            permitted_top_failure_list.append((strat_unit_name, 'No base'))
        
        #TODO This might be redundant and can be deleted
        #If the top was not in the list of permitted tops (this should have
        #been filtered out beforehand...) continue to the next top                                
        else:
            continue
            
    permitted_top_checker_result_dict = {}
    permitted_top_checker_result_dict['Success'] = permitted_top_success_list
    permitted_top_checker_result_dict['Failure'] = permitted_top_failure_list
    
    return permitted_top_checker_result_dict



def create_well_tops_dict(raw_formation_tops_info, uwi_column='UWI'):
    """Create a dictionary with all of the formation top information,
    organized by UWI"""
    
    well_column_data = raw_formation_tops_info[uwi_column]
    unique_wells = get_unique_wells(well_column_data)
    
    well_tops_dict = {}
    
    for uwi in unique_wells:
        well_specific_data = raw_formation_tops_info[raw_formation_tops_info['UWI'] == uwi]
        
        well_tops_dict[uwi] = well_specific_data
        
    
    return well_tops_dict
        

def create_verified_success_well_tops_dict(tip_tops_df, 
                                           permitted_tops_and_bases_filename):
                                    
    """The well tops must have already been put through the top_interpreter_pick
    (tip) routine"""
    
    #Get tops data into dictionary with UWI: (Df with tops info) structure
    tip_well_tops_dict = create_well_tops_dict(tip_tops_df)
    
    #Load the permitted tops and bases
    permitted_tops_and_bases_dict = load_permitted_tops_and_bases(permitted_tops_and_bases_filename)
    
    #Instantiate dictionary to store verified success tops by uwi
    consolidated_verified_success_tops_dict = {}
    
    #Loop over wells in tip_well_tops_dict
    for uwi, tip_well_tops_df in tip_well_tops_dict.items():
               
        #Check tops in the well and return permitted tops dictionary (This
        #dictionary has a 'Success' key for tops that were successfully
        #verified and a 'Failure' key for tops that were did not have a top
        #and base
        permitted_tops_for_well_dict = permitted_top_checker(tip_well_tops_df, 
                                                        permitted_tops_and_bases_dict)
        
        success_permitted_tops_for_well = permitted_tops_for_well_dict['Success']
        
        consolidated_verified_success_tops_dict[uwi] = success_permitted_tops_for_well
    
    return consolidated_verified_success_tops_dict
    
    
    
    


def load_consolidated_formation_tops_info(formation_top_fn):
            
    raw_formation_tops_info = pd.read_csv(formation_top_fn)
    
    return raw_formation_tops_info




def get_success_tops_dict(tops_df,
                          permitted_tops_and_bases_filename,
                          interpreter_preference_list):
    
    top_interpreter_pick_df = top_interpreter_pick(tops_df, 
                                                   permitted_tops_and_bases_filename, 
                                                   interpreter_preference_list)

    success_tops_dict = create_verified_success_well_tops_dict(top_interpreter_pick_df,
                                                               permitted_tops_and_bases_filename)  
    
    return success_tops_dict
    

def alias_well_tops(tops_df, alias_table_fn):
    """
    Replace well tops in tops df DataFrame with aliased version
    

    Parameters
    ----------
    tops_df : DataFrame
        DataFrame of well tops.
    alias_table_fn : str
        Path to alias table information.

    Returns
    -------
    merged_tops_df : DataFrame
        DataFrame with aliased tops.

    """
    #Read alias table
    alias_table = pd.read_csv(alias_table_fn, names=['Pick Name_Raw', 'Pick Name'], skiprows=1)
    
    #Perform left merge between tops_df and alias_table, and efficient way
    #to assign alias names to the dataframe
    merged_tops_df = tops_df.merge(alias_table, on='Pick Name_Raw', how='left')
        
    
    return merged_tops_df

#TODO This function under development
def apply_formation_tops_well_log_dict(well_log_dict, 
                                       tops_df_fn, permitted_tops_and_bases_fn,
                                       interpreter_preference_list, alias_table_fn=None):
    """
    Take a dictionary of well log objects and apply formation top information
    to each log object in the dictionary.

    Parameters
    ----------
    well_log_dict : dict
        Dictinary of log objects.
    tops_df_filename : TYPE
        Filename for tops information.
    permitted_tops_and_bases_filename : TYPE
        Filename for top and base checking.
    interpreter_preference_list : list
        List of interpreter preference, from most preferred to least preferred.

    Returns
    -------
    well_log_dict : dict
        Dictionary with well log information applied.

    """
    
    #Load tops dataframe
    tops_df = pd.read_csv(tops_df_fn)
    
    #If an alias table filename has been provided, perform aliasing through
    #alias_well_tops()
    if alias_table_fn != None:
        tops_df = alias_well_tops(tops_df, alias_table_fn)
        
        
    success_tops_dict = get_success_tops_dict(tops_df, permitted_tops_and_bases_fn, interpreter_preference_list)
    
    for uwi, well_log in well_log_dict.items():
        try: 
            success_tops_well = success_tops_dict[uwi]
        except KeyError:
            print(f'{uwi} not found in Success Tops Dictionary')
            continue
        
        add_strat_unit_column(well_log, success_tops_well)       
    
    return well_log_dict



if __name__ == "__main__":
    
    with open(r"C:\Programming\AI_Production_Log_Project_2023\Scripting Session Folders\22APR2024 Bypassed Pay Session\test_mini_log_dictionary.pkl", 'rb') as f:
        well_logs_dict = pkl.load(f)
        
    tops_df_filename = r"C:\Programming\AI_Production_Log_Project_2023\Scripting Session Folders\22APR2024 Bypassed Pay Session\Tops\Bypassed_Pay_AI_Tops_18APR2024_replaced.csv"

    permitted_tops_and_bases_filename = r"C:\Programming\AI_Production_Log_Project_2023\Scripting Session Folders\22APR2024 Bypassed Pay Session\Tops\kingdom_lithostrat_tops_and_bases 17APR2024.csv"
    
    interpreter_preference_list = ['GAF', 'PXT', 'RGH', 'XRA']   
        
    apply_formation_tops_well_log_dict(well_logs_dict, tops_df_filename, permitted_tops_and_bases_filename, interpreter_preference_list)
    
    
    