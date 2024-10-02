# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:12:58 2019

@author: Gfoo
"""

import datetime
import pandas as pd
import ast
import pdb
from . import plastic as pl

WELL_NAME = 'WELL'
EFFECTIVE_DATE = 'EFFECTIVE DATE'
PERF_INTERVAL_TOP = 'MD TOP'
PERF_INTERVAL_BASE = 'MD BASE'
STATUS = 'STATUS'


#Enter filepath for the perf data spreadsheet
perf_data_filepath = r"C:\Programming\AI_Production_Log_Project_2023\Scripting Session Folders\10JUN2024 Scripting Session\Perfs\Parex_openwells_perf_export_09SEP2024.xlsx"

#Load into dataframe
perf_data = pd.read_excel(perf_data_filepath, 'Data', parse_dates=['EFFECTIVE DATE'])

#Get list of wells by taking well column, dropping duplicates, converting to
#list and sorting alphabetically
well_column_name = 'WELL'
wells = perf_data[well_column_name].drop_duplicates().tolist()
wells.sort()

perf_data_list = []

#Filename for CSV export of final data set


def alter_perf_record(status, perf_top, perf_bottom, open_perfs, closed_perfs):
    """This is a function that alters the open_perfs and closed_perfs lists.
    Subsequently in the make_records function, these lists are used to write
    the daily records of open and closed perf intervals"""
    #Create a string based on top and bottom of perf interval    
    perf_interval = ('{}-{}').format(perf_top, perf_bottom)
    
    # Check the status property. If 'OPEN', add to the open perfs list and
    # check to see if it's in the closed perfs list. If it is, remove this
    # interval from the closed perfs list
    if status == 'OPEN':
        #print 'OPEN PERFS'
        open_perfs.append(perf_interval)
        if perf_interval in closed_perfs:
            closed_perfs.remove(perf_interval)
    # If the status is 'CLOSED', try to remove this interval from the
    # open_perfs list. If the interval is not there (which raises a value
    # error), add the interval to the closed_perfs list. This exception
    # catching is in place because of some duplicate entries for 'CLOSED'.
    # E.g. the additional 'CLOSED' entry is added with a comment 'ABANDONED'
    # to signal that the well has been abandoned
    if status == 'CLOSED':
        #print 'CLOSED PERFS'
        try:
            open_perfs.remove(perf_interval)
        except(ValueError):
            pass
        if perf_interval not in closed_perfs:
            closed_perfs.append(perf_interval)
            
    return (open_perfs, closed_perfs)
    
#This function converts a string representation of a list back to a list type
def string_to_list(string):
    str1 = ast.literal_eval(string)
    return str1

def make_records(well_name, start_date, end_date, open_perfs_val, closed_perfs_val):
    """This function creates entries that show the open perforations and
    closed perforations for each day for each well from the date of the first
    entry in the perfs table corresponding to that well, to today's date."""
    
    #Initialize the date counter on the date corresponding to the perf entries
    date_counter = start_date
    # While the counter date is before the end date of the next perf event,
    # append the perf_data_list list with entries that show the well, date
    # open perforations and closed perforations corresponding to that date
    while  date_counter < end_date:
       ###This expression below does not work, copies values of open_perfs and closed_perfs from last date of well. Why?
#       entry = [well_name, date_counter, open_perfs_val, closed_perfs_val]
        entry = [well_name, date_counter, string_to_list(open_perfs_val.__repr__()), string_to_list(closed_perfs_val.__repr__())]
        perf_data_list.append(entry)
       
        date_counter = date_counter + datetime.timedelta(days=1)
        
def get_last_date_data(all_data_df, last_date=0):
   
    last_entry_list = []
    
    well_names = list(set(list(all_data_df['Well_name'])))
    for well in well_names:
        well_df = all_data_df[all_data_df['Well_name'] == well]
        last_entry = well_df.iloc[[-1]]
        last_entry_list.append(last_entry)
    
    last_entry_df = pd.concat(last_entry_list)
    
    return last_entry_df

def check_names(well_list, df_compare):
    results_list = []
    for well in well_list:
        result = False
        if well in df_compare['ANH_WELL_NAME'].values:
            result = True
            tag = 'ANH_WELL_NAME'

        elif well in df_compare['COREX_WELL_NAME'].values:
            result = True
            tag = 'COREX_WELL_NAME'           
            
            
        if result == True:
            series = df_compare[tag]
            match = series == well
            index = series[match].index
            print(index)
            
            #This is the way to extract just the string value for UWI.
            UWI = df_compare.loc[index, 'UWI'].values[0]
            results_list.append((well, UWI, tag))
        else:
            results_list.append((well, 'No match', 'NA'))
    
    results_df = pd.DataFrame(results_list, columns=['Well name', 'UWI', 'Source'])
    
    return results_df




def main():
    for well in wells:
        #Instantiate new, empty lists for open perfs and closed perfs for each well
        open_perfs = []
        closed_perfs = []
        
        # Filter dataframe to just the data corresponding to that well
        well_data = perf_data[perf_data[WELL_NAME] == well]
        #Obtain a list of dates of the mechanical status changes in each well
        #by copying the date, column, sorting it from oldest to newest, and
        #dropping the duplicate entries
        well_data = well_data.sort_values(EFFECTIVE_DATE)
        well_data_dates = well_data[EFFECTIVE_DATE].drop_duplicates().tolist()
        
        #In this particular well, for each unique date that is in the perfs table,
        #filter the data corresponding to that date
        for i in range(len(well_data_dates)):
            well_data_date = well_data_dates[i] #Index well data dates and store date
            try:
                #Ensure that the data type is valid
                assert(type(well_data_date) != pd.NaT)
            except(AssertionError):
                #print if data type does not conform and exit for loop
                print('Error in well {}'.format(well))
                break
            
            #Filter well_data dataframe further to only include the date that is
            #being indexed
            date_data = well_data[well_data[EFFECTIVE_DATE] == well_data_date]
            #What is the start of the period where these perf intervals are valid?
            start_date = well_data_date        
            #If the date is not the last one corresponding to the well, set the end
            #date of this period to the next entry in well_data_dates
            if well_data_dates[i] != well_data_dates[-1]:
                end_date = well_data_dates[i+1] - datetime.timedelta(days=1)
            #If the date is the last one corresponding to the well, set the date
            #of the end of the period to today.
            else:
                end_date = pd.Timestamp(datetime.date.today())
                
            #Get the number of entries in the perf table that correspond to that
            #particular well and that particlar date
            date_rows_length = date_data.shape[0]
            
            #Iterate over all of the dates that correspond to that well in the
            #perf table, and pass the data to the alter_perf_record function
            for i in range(date_rows_length):
                entry = date_data.iloc[i]
                status = entry[STATUS]
                perf_top = entry[PERF_INTERVAL_TOP]
                perf_bottom = entry[PERF_INTERVAL_BASE]
                open_perfs, closed_perfs = alter_perf_record(status, perf_top, perf_bottom, open_perfs, closed_perfs)
    
            #Now that the open_perfs and closed_perfs lists have been altered for
            #that given well on that given date, send the data to the make_records
            #function so that the daily entries can be generated
            make_records(well, start_date, end_date, open_perfs, closed_perfs)
    
    #Convert the list with the daily perf interval entries to a dataframe for easier handling
    #Make date a timestamp data type, and the perfs a list datatype
    perf_data_df = pd.DataFrame(perf_data_list, columns=['Well_name', 'Date', 'Open_Perfs', 'Closed_Perfs'])
    
    return perf_data_df





def create_perfs_dict(df, well_column='UWI'):
    """
    Take a dataframe of well mechanical statuses and convert this into a
    perf dict

    Parameters
    ----------
    df : DataFrame
        The dataframe with the mechanical status (open perf and closed perf)
        information.

    Returns
    -------
    perf_dict : dict
        A dictionary in the form {UWI: (open_perfs, closed_perfs)}

    """
    
    perfs_dict = {}
    
    well_list = list(set(list(df[well_column])))
    
    for well in well_list:
        well_df = df[df[well_column] == well]
        
        open_perfs = well_df['Open_Perfs']
        closed_perfs = well_df['Closed_Perfs']
        
        perfs_dict[well] = (open_perfs, closed_perfs)
    
    return perfs_dict
                     
    
    
    

def assign_perfs_to_df(df, perfs_dict):
    """
    Parameters
    ----------
    df : DataFrame
        The dataframe with the ASCII curve data upon which the perf information
        will be assigned.
    perfs_dict : dict
        A dictionary in the form {UWI: (open_perfs, closed_perfs)}

    Returns
    -------
    df_with_perfs : DataFrame
        The original dataframe with perforation information applied.

    """
    # def add_strat_unit_column(well_object, permitted_top_success_list, 
    #                           well_name_index='Well_name',
    #                           depth_index='DEPT', 
    #                           strat_unit_name='Strat_unit_name'):
    open_perf_column_mega_df = 'Perf_open'
    closed_perf_column_mega_df = 'Perf_closed'
    
    
   
    for well_name, perfs in perfs_dict.items():
        open_perfs, closed_perfs = perfs
        
        #Get well names from log data df
        well_names_in_df = list(set(list(df.index.get_level_values('Well_name'))))
        
        if well_name in well_names_in_df:                            
        
            well_data_df = df[df.index.get_level_values('Well_name') == well_name]
            
            def _verify_depths(well_data_df, raw_top, raw_base):
                #Put the perforation top and base through Plastic's find closest depth
                #method to ensure that there will be valid indicies for the slice
                
                # _verified_top, verified_top_index = pl.find_closest_depth_value(well_data_df, raw_top)
                # _verified_base, verified_base_index = pl.find_closest_depth_value(well_data_df, raw_base)
                
                # return (verified_top_index, verified_base_index)
                
                verified_top_depth, _verified_top_index = pl.find_closest_depth_value(well_data_df, raw_top)
                verified_base_depth, _verified_base_index = pl.find_closest_depth_value(well_data_df, raw_base)
                
                return (verified_top_depth, verified_base_depth)
            
            # def __apply_perfs(df, column_integer_location, verified_top_index, verified_base_index):
            #     #Apply the perfs to the mega_df
            #     df.iloc[verified_top_index:verified_base_index,
            #                       column_integer_location] = 1
                
            def _apply_perfs(df, well_name, column_name, verified_top_depth, verified_base_depth):
                #Apply the perfs to the mega_df
                df.loc[(well_name, verified_top_depth):(well_name, verified_base_depth),
                                  column_name] = 1
                
                #return df
            
            
            
            #If there are entries for open_perfs, 
            if len(open_perfs) > 0:

                #Need to index at 0 to get the values out of the series
                for perf_cluster in open_perfs.values[0]:
                    top_depth_str, base_depth_str = perf_cluster.split('-')
                    try:
                        top_depth = float(top_depth_str)
                        base_depth = float(base_depth_str)
                        
                        assert(base_depth > top_depth)
                        
                    except TypeError:
                        print(f'There was a type error in open perfs. Top depth {top_depth}, Base depth {base_depth}')

                    except AssertionError:
                        print(f'The base depth was not below the top depth in {well_name}. Top depth {top_depth} Base depth {base_depth}')
                        top_depth = base_depth
                        base_depth = top_depth

                    
                    #TODO This might not work, getting the indicies from the indexed dataframe and trying to pass it to the mega_df
                    verified_top_depth, verified_base_depth = _verify_depths(well_data_df, top_depth, base_depth)
                    
                    # #Get integer column location of open perfs
                    # open_perfs_int_loc = well_data_df.columns.get_loc('Perf_open')
                    
                    #_apply_perfs(well_data_df, open_perfs_int_loc, verified_top_index, verified_base_index)
                    _apply_perfs(df, well_name, open_perf_column_mega_df, verified_top_depth, verified_base_depth)
                    #df = _apply_perfs(df, 'Open_Perfs', verified_top_index, verified_base_index)
                    
            if len(closed_perfs) > 0:
                #Need to index at 0 to get the values out of the series
                
                for perf_cluster in closed_perfs.values[0]:
                    top_depth_str, base_depth_str = perf_cluster.split('-')
                    try:
                        top_depth = float(top_depth_str)
                        base_depth = float(base_depth_str)
                        
                        assert(base_depth > top_depth)
                        
                    except TypeError as e:
                        msg = e.args[0]
                        print(f'There was a type error in closed perfs. Top depth {top_depth}, Base depth {base_depth}')
                        print(msg)
                    except AssertionError as e:
                        print(f'Assertion Error {e}')
                        msg = e.args[0]
                        print('The base depth was not below the top depth')
                        print(msg)
                    
                    #TODO This might not work, getting the indicies from the indexed dataframe and trying to pass it to the mega_df
                    verified_top_depth, verified_base_depth = _verify_depths(well_data_df, top_depth, base_depth)
                    
                    # #Get integer column location of closed perfs
                    # closed_perfs_int_loc = well_data_df.columns.get_loc('Perf_closed')
                    
                    _apply_perfs(df, well_name, closed_perf_column_mega_df, verified_top_depth, verified_base_depth)
                    #df = _apply_perfs(df, 'Closed_Perfs', verified_top_index, verified_base_index)
        else:
            continue
        
    return df
                
                
if __name__ == '__main__':
    perf_data_df = main()
#Get last date data
#last_date_data = get_last_date_data(perf_data_df, last_date=0)

##Write last_date_data to csv
#last_date_export_fn = 'PXT_last_date_perf_data.csv'
#last_date_data.to_csv(last_date_export_fn)

##Write Dataframe to CSV file for export
#export_filename = 'Perf_Interval_Data_pxtwells.csv'
#perf_data_df.to_csv(export_filename)       
        
        
 


            
            
        