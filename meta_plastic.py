# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 21:51:58 2017

@author: Gordon Foo
"""

"""Meta Plastic

UPDATED 16FEB2023
UPDATED 13FEB2024 to replace scipy.misc.imread with imageio library
UPDATED 14FEB2024 to turn __name__ == __main__ functionality into a method called super_df()
    The plastic module was also changed on this date to give the ASCII log data database 


This file is in need of cleaning, but incorporates a lot of functionality

The main functionality is to take a directory where log data is stored, and
to turn the files within into log class objects that are defined in the plastic module. When processing
multiple log class objects through create_logs(), a dictionary with well names as the keys is returned.

The module also has legacy functionality to filter a large set of log data by formation,
create histograms of large sets of log data, map sets of amalgamated log data,
extract log data based on mneumonic, and plot this data on charts. 

"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imageio import imread
import numpy as np
import pdb
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.colors as mcolors

from . import plastic as pl

L1 = []

logs_dir = r'C:\Programming\AI_Production_Log_Project_2023\Scripting Session Folders\14FEB2024 Bypassed Pay Session\Logs'
#tops_fn = "C:\Users\gfoo\Desktop\Programming\Log_reader(plastic)\SCasanareCoreTops_GAF_Only 13NOV2017.csv" ##Commented out 13FEB2024
#coordinate_filename = r"C:\Users\gfoo\Desktop\Programming\Log_reader(plastic)\GGX_XYcoords_15NOV2017.csv"  ##Commented out 13FEB2024
#map_filename = r"C:\Users\gfoo\Desktop\Programming\Log_reader(plastic)\LLA34_map.jpg"
#coords_fn - 

tops_fn = 'Exported_Central_Llanos_Formation_Tops 16FEB2024.csv'

def create_logs(logs_dir, tops_fn=0):
    """The log dict will store all of the log objects produced by the script in
    a dictionary, where the key is the file name, minus the extension, and the
    value is the corresponding log object"""
    log_dict = {}
    log_init_list = get_initiation_list(logs_dir) #Get filenames to create log objects
    for well_name, file_name in log_init_list:
        print (well_name)
        log = pl.create_log(file_name, well_name, 0) #Zero is BS filler argument
        log_dict[well_name] = log
    
    if tops_fn != 0:
        for well_name, log in log_dict.items():
            print (well_name)
            log.assign_formation_tops()
            #This DELTA_PHIN_PHID below is optional and can be removed
            log.add_delta_PHIN_PHID()
            #log.assign_xy_coord(coordinate_filename) ##Commented out 13FEB2024
    return log_dict

def get_initiation_list(logs_dir):
    """Get list of file names and format a string to concatenate the root
    directory with the file name. Use list comprehension to loop over each
    file name that comes out of the list produced by os.listdir()
    """
    log_file_names = [('{}\\{}').format(logs_dir, f_name) for f_name in os.listdir(logs_dir)]
    #The log name is a slice of the filename, which removes the file extension
    log_names = [i.split('\\')[-1][:-4] for i in log_file_names]
    #Zipping the log names and log file names gives a list of tuples that
    #can be looped over to create the log objects in another function
    log_initiation_list = zip(log_names, log_file_names)
    
    return log_initiation_list

#TODO Dangerous code with eval()
def filter_data(super_df, conditions_string):
    """Take the string that was used to describe the conditions and parse it into
    individual arguments (which are still strings)

    Parameters
    ----------
    super_df : Dataframe
        DataFrame with consolidated data
    conditions_string : str
        String of conditions

    Returns
    -------
    filtered_df, a dataframe with the specified conditions applied

    """

    conditions = parse_conditions(conditions_string)
    s = ""
    #If there is only one condition, no need to loop. The fancy index statement
    #is constructed as a string and then turned into live code by eval()
    if len(conditions) == 1:
        condition = ('super_df{}').format(conditions[0])
        filtered_df = super_df[eval(condition)]
    #If there are multiple conditions, concatenate each condition together with 
    #an ampersand and then add last condition without ampersand
    else:
        for condition in conditions[:-1]:
            s+= ('(super_df{}) & ').format(condition)
        s += ('(super_df{})').format(conditions[-1])
        #Fancy index the super data frame with the conditions, which are turned
        #into live code by eval()
        filtered_df = super_df[eval(s)]   
    
    return filtered_df

def parse_conditions(condition_string):
    
    """
    Split the condition string on commas, into individual conditions
    """
    condition_list = condition_string.split(',')
    #Split each condition on space, and strip off the white space for each of
    #these elements of the condition
    condition_split_list = [i.strip().split(' ', 2) for i in condition_list]
    
    #condition_split_list is a list of list. Format each condition (which is an
    #element of the parent list) into the proper string for indexing, and use
    #list comp. to loop over each condition (element in the parent list). The
    #individual elements of each condition are strings, and are indexed i[0], 
    #i[1], i[2] to be combined into the proper string in the list
    
    final_conditions =[("['{}'] {} {}").format(i[0], i[1], i[2]) for i in condition_split_list]
    
    return final_conditions

def count_filtered_data(filt_data_df, logs_dict):
    #Group the filtered data by the well name. Then, get the counts of each
    #column. Transpose this dataframe which now has the counts, and calculate
    #the maximum count of each column (each column corresponds to a well).
    #Store the well name with the maximum count as a dictionary max_count_dict
    max_count_dict = filt_data_df.groupby('Well_name').count().T.max().to_dict()
    #Initialize dictionary which will be used to store data on the depth
    #thicknesses that correspond to each row count.
    #For each well, depth thickness = row count * step
    depth_total_dict = {}
    for well_name, row_count in max_count_dict.items():
        #logs_dict is the dictionary that stores all of the log objects. If the
        #name of the well matches the UWI of a well object in the dictionary,
        #return the step of that log object and multiply it by the row count to
        #get the depth thickness. Save the well name and thickness key/value
        #pair in the new dictionary depth_total_dict()
        for well_uwi, log_object in logs_dict.items():
            #If the UWI is the same as the well name from the max_count_dict,
            #then multiply the count value by the step value for that part-
            #icular log, which will yield the depth thickness for that well
            if log_object.W.UWI.value == well_name:
                step = float(log_object.W.STEP.value)
                depth_thickness = max_count_dict[well_name] * step
                depth_total_dict[well_name] = depth_thickness
    return depth_total_dict

#Create a bar chart with the results of the filtered data dict
def vert_bar_chart(d):
    getKey = lambda x: x[1]
    L1 = sorted(d.items(), key=getKey, reverse=True)
    labels = [i[0] for i in L1]
    values = [i[1] for i in L1]
    plt.bar(range(len(L1)), values, align='center')
    plt.xticks(range(len(d)), labels, rotation ='vertical')
    
#def assign_XY
    


def plot_mean_histogram_data(dataframe, curve_name, bins=20):
    series1 = dataframe.groupby('Well_name').mean()[curve_name]
    clean_data = remove_nans_list(series1.tolist())
    
    return plt.hist(clean_data, bins)

def plot_mean_ordered_data(dataframe, curve_name):
    dict1 = dataframe.groupby('Well_name').mean()[curve_name].todict()
    dict2 = remove_nans_dict(dict1)
    
    return vert_bar_chart(dict2)
    
def remove_nans_list(list):
    return [x for x in list if str(x) != 'nan']

def remove_nans_dict(d):
    return {key: value for key, value in d.items() if str(value) != 'nan'}

def get_buff_map_coord_lims(axis_limits, pcg_buffer):
    """Take the limits of the imported image, and the data that will be
    used for the map, apply the buffer (in percentage) to either side of the
    map data and return these coordinates"""
    #Unpack min and max values for the the particular axis of the
    #map that is being evaluated (i.e. x axis or y axis)
    axis_min, axis_max = axis_limits
    #Calculate the range of the map data by subtracted in the min from the max
    axis_range = axis_max - axis_min
    #The amount of buffer to use for that axis, in map coordinate units, will
    #be the percentage buffer times the range
    buffer_in_map_coords = axis_range * pcg_buffer
    #Apply the buffer in map coordinates to the max and min
    buf_axis_min = axis_min - buffer_in_map_coords
    buf_axis_max = axis_max + buffer_in_map_coords
    
    print ('buf_axis_min {} buf_axis_max {}').format(buf_axis_min, buf_axis_max)
   
    return buf_axis_min, buf_axis_max
    

def get_map_plot(filename, map_lims, img_lims):
    """ In this function 'map' refers to the map that will eventually
    be displayed, and image refers to the large, base image that is being
    imported and sliced"""
    img_buffer = 0.1
    #Read image to array
    raw_img = imread(filename)
    
    def get_slice_idx(m_min, m_max, i_min, i_max, px_dim, axis):
        """This function accepts the map data max and min and the image max
        and min and uses it to calculate where to take the slice of the map
        in terms of row, column location in the imported image. 'px_dim
        is the pixel length of the axis being calculated"""
        #Get the range of the image in map coordinates by 
        i_range = i_max - i_min
        buf_min, buf_max = get_buff_map_coord_lims((m_min, m_max), img_buffer)
        
#        start = int((((buf_min - i_min) / i_range)) * px_dim)
#        end = int((((buf_max - m_min) / i_range)) * px_dim)
        
        if axis == 'x':
            start = int(((buf_min - i_min) / i_range) * px_dim)
            end = int(((buf_max - i_min) / i_range) * px_dim)
        elif axis == 'y':
            start = int(((i_max- buf_max) / i_range) * px_dim)
            end = int(((i_max - buf_min) / i_range) * px_dim)

                 
        print ('Start {}, End {}').format(start, end)
        
        return (start, end), (buf_min, buf_max)
        
   
    
  
    #Unpack map limits to local variables. map_lims is a tuple with len 4
    map_xmin, map_xmax, map_ymin, map_ymax = map(float, map_lims)    
    print ('map_xmin {}, map_xmax {}, map_ymin {}, map_ymax {}').format(map_xmin, map_xmax, map_ymin, map_ymax)
    
    #Unpack image limits to local variables. img_lims is a tuple with len 4
    img_xmin, img_xmax, img_ymin, img_ymax  = map(float, img_lims)
    
    print ('img_xmin {}, img_xmax {}, img_ymin {}, img_ymax {}').format(img_xmin, img_xmax, img_ymin, img_ymax)
    #Get the number of rows and columns in the image by unpacking raw_img.shape
    y_len_px, x_len_px, px_val = raw_img.shape
    print ('raw_img.shape {}').format(raw_img.shape)
    #Get indicies, buffered map units for x-axis
    x_idx, bx = get_slice_idx(map_xmin, map_xmax, img_xmin, img_xmax, x_len_px, 'x')
    #Get indicies, buffered may units for y-axis
    y_idx, by = get_slice_idx(map_ymin, map_ymax, img_ymin, img_ymax, y_len_px, 'y')
    #Get the map plot by slicing row, column (convention for images)    
    map_plot = raw_img[y_idx[0]:y_idx[1], x_idx[0]:x_idx[1],:]
    print ('y_idx[0] = {} y_idx[1] = {} x_idx[0] = {} x_idx[1] = {}').format(y_idx[0],y_idx[1],x_idx[0],x_idx[1])
    
    extents = (bx,by)
    
    return (map_plot, extents)
    
    
    
   
#    x_slice_idx = get_slice_indicies(x_params)    
#    y_slice_idx = get_slice_indicies(y_params)
#    
#    map_plot = raw_image[y_slice_idx[0]:y_slice_idx[1], 
#                         x_slice_idx[0]:x_slice_idx[1],:]
#    map_plot_limits = get_buffered_map_plot_limits()
#    
#    return (map_plot, map_plot_limits)
    
    
def create_data_map(counted_data, logs_dict, map_filename):
    d1 = {get_xy_coords(well_name, logs_dict): value for well_name, 
          value in counted_data.items() if None not in 
          get_xy_coords(well_name, logs_dict)}
#    print d1.keys()
    x_values = [x[0] for x in d1.keys()]
    y_values = [x[1] for x in d1.keys()]
    
    map_limits = (np.min(x_values), np.max(x_values), np.min(y_values), np.max(y_values))
    img_limits = (1136000, 1193000, 963000, 1019000)
    size = d1.values()
    
    map_plot, m_lims = get_map_plot(map_filename, map_limits, img_limits)
    fig, ax = plt.subplots()
    extents = [lim for i in m_lims for lim in i]
    print ('Extents are {}').format(extents)
    #Extent : scalars (left, right, bottom, top)
    ax.imshow(map_plot, extent=extents)
    ax.scatter(x=x_values, y=y_values, s=size, color = 'white')
    
    return  ax

def get_xy_coords(well_name, logs_dict):
    for well_uwi, log_object in logs_dict.items():
        if log_object.W.UWI.value == well_name:
            return (log_object.W.XPOS, log_object.W.YPOS)
            
def crossplot_data(df, x_var, y_var):
    df_xy = df[[x_var, y_var]].dropna()
    x_list = df_xy[x_var].tolist()
    y_list = df_xy[y_var].tolist()
    fig, ax = plt.subplots()
    ax.scatter(x=x_list, y=y_list)
    if 'Res' in x_var:
        ax.set_xscale('log')
    if 'Res' in y_var:
        ax.set_yscale('log')
    
    return ax
    
def round_log_data(data_list, step):
    return [((round(i/step))*step) for i in data_list]
            

def create_super_df(logs_dict):
    """Create a super dataframe from a dictionary of logs"""
    super_list = [i.A for i in logs_dict.values()] #Use list comprehension to extract ASCII data

    
    super_df = pd.concat(super_list)
    return super_df


def create_log_data_dict_from_super_df(super_df):
    """Create a dictionary with UWI: ASCII log data entries based on a 
    concatenated super dataframe"""
    #List of log UWIs
    UWI_list = list(set(super_df.index.get_level_values(0)))
    #Sort alphabetically
    UWI_list.sort()
    
    #Create groupby object
    groupby_super_df = super_df.groupby(level=0)
    
    log_data_dict = {i: groupby_super_df.get_group(i) for i in UWI_list}
    
    return log_data_dict

def filter_super_df_for_wells_with_vals_in_col(super_df, column_name, value_to_screen_out=np.nan):
    """Take a super df and only leave the wells that have particular values
    in a column. This workflow can be used to filter out wells that do not
    have data in a particular column. The complete dataframe for the wells
    that do have data is left intact"""
    
    #Get rid of nan rows
    super_df_pre_screen = super_df.dropna(subset=[column_name])
    
    #Create a list of wells from the remaining dataframe that does not have
    #rows where the column value is nan
    super_df_wells = list(set(super_df_pre_screen.index.get_level_values('Well_name')))
    
    #Get an index on the super_df as the first step in creating a mask
    super_df_well_index = super_df.index.get_level_values('Well_name')
    
    #Create mask with index from previous index, plus the isin function to
    #only take the rows where the well name is in the wuper_df_wells list
    well_name_mask = super_df_well_index.isin(super_df_wells)
    
    #Apply mask as final step
    super_df_filtered = super_df[well_name_mask]

    return super_df_filtered        
    
def apply_XY_coords_to_dict(
                            well_log_dict, 
                            consolidated_well_info_fn = r"C:\Programming\AI_Production_Log_Project_2023\Scripting Session Folders\22APR2024 Bypassed Pay Session\Consolidated_Well_Info 08AUG2024.xlsx"
                            ):
    """For a dictionary of logs, or a dictionary of ASCII data dataframes,
    lookup the associated coordinate values from the consolidated well info
    spreadsheet and apply a column to the respective dataframe with x and y
    coordinates
    """
    
    consolidated_well_info_df = pd.read_excel(consolidated_well_info_fn)
    
    consolidated_well_info_df =  consolidated_well_info_df.set_index('UWI')
    
      
    for UWI, data in well_log_dict.items():
        try:
            assert type(data) == pl.Log
            well_data_df = data.A            
        except AssertionError:
            try:
                assert type(data) == pd.core.frame.DataFrame
                well_data_df = data
            except AssertionError:
                data_type = type(data)
                print(f'Data type {data_type} is not valid')
        
        #The column selected has calculated Magna-Sirgas Origen Bogota XY
        #values derived from the WGS83 Lat-Long coordinates
        consolidated_df_x_coord_column = 'XCOORD_MAGNA_SIRGAS_BOGOTA_PXT'
        
        consolidated_df_y_coord_column = 'YCOORD_MAGNA_SIRGAS_BOGOTA_PXT'
        
        #consolidated_df_coord_system_column = 'DATUM'
        
        
        try:
            well_x_coord = consolidated_well_info_df.loc[UWI][consolidated_df_x_coord_column]
            well_y_coord = consolidated_well_info_df.loc[UWI][consolidated_df_y_coord_column]
            #well_coord_system = consolidated_well_info_df.loc[UWI][consolidated_df_coord_system_column]
        except KeyError:
            print(f'Well {UWI} not found')
            continue
        
        well_data_df['X_COORD'] = well_x_coord
        well_data_df['Y_COORD'] = well_y_coord
        #well_data_df['COORD_SYSTEM'] = well_coord_system
        
    return well_log_dict

class Plot:
    def __init__(self):
        pass
    
    # Function to drop columns with only NaN values and create the pairplot
    def hex_pair_plot(df1, df2=None):
        # Identify columns with all NaN values in either df1 or df2
        if df2 is not None:
            # Union of columns with all NaNs in either dataframe
            cols_to_drop_df1 = df1.columns[df1.isna().all()]
            cols_to_drop_df2 = df2.columns[df2.isna().all()]
            cols_to_drop = cols_to_drop_df1.union(cols_to_drop_df2)
            
            # Drop the identified columns from both dataframes
            df1 = df1.drop(columns=cols_to_drop)
            df2 = df2.drop(columns=cols_to_drop)
    
            # Calculate overall min and max for each column considering both dataframes
            overall_min = pd.concat([df1.min(), df2.min()], axis=1).min(axis=1)
            overall_max = pd.concat([df1.max(), df2.max()], axis=1).max(axis=1)
        else:
            df1 = df1.dropna(axis=1, how='all')
            overall_min = df1.min()
            overall_max = df1.max()
    
    
        def hexbin_(x, y, gridsize=30, **kwargs):
            # Drop NaN values only for the pair of columns being plotted
            x = x.dropna()
            y = y.dropna()
    
            # Align x and y by their indices after dropping NaNs
            valid_indices = x.index.intersection(y.index)
            x = x.loc[valid_indices]
            y = y.loc[valid_indices]
    
            # Use the pre-calculated overall min and max values
            x_min, x_max = overall_min[x.name], overall_max[x.name]
            y_min, y_max = overall_min[y.name], overall_max[y.name]
            
            # Print debug info
            print(f"Plotting {x.name} vs {y.name}")
            print(f"Using x range: {x_min} to {x_max}")
            print(f"Using y range: {y_min} to {y_max}")
    
            plt.hexbin(x, y, gridsize=gridsize, cmap='Blues', extent=(x_min, x_max, y_min, y_max), **kwargs)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            
            # Overlay second dataset if provided
            if df2 is not None:
                x2, y2 = df2[x.name].dropna(), df2[y.name].dropna()
                valid_indices2 = x2.index.intersection(y2.index)
                x2 = x2.loc[valid_indices2]
                y2 = y2.loc[valid_indices2]
                
                # Use the same overall min and max for the overlay data
                plt.hexbin(x2, y2, gridsize=gridsize, cmap='Reds', alpha=0.5, extent=(x_min, x_max, y_min, y_max), **kwargs)
    
        # Create PairGrid without forcing aspect ratio
        g = sns.PairGrid(df1)
        
        # pdb.set_trace()
        
        # Apply hexbin to the upper and lower triangles
        g.map_upper(hexbin_)
        g.map_lower(hexbin_)
        g.map_diag(plt.hist)  # Keep the histograms on the diagonal
        
        columns = g.x_vars
        
        # Ensure consistent axis limits across all subplots
        for ax in g.axes.flatten():
            ax.autoscale(True)
    
        # Show the plot
        
        
        plt.show()
        
        
        print(columns)
        print(overall_min)
        print(overall_max)
        
        return (columns, overall_min, overall_max)
    
    def create_gradational_flag_curve(df, criteria_dict, formation_pen_tup=None):
               
        """
        Assign point values based on how much the row deviates from the norm?
        
        Based on a dictionary of criteria, assign a point value based on how
        many of the key criteria for pay are met. Maximum score is 1
        
        Dictionary structure is {curve_name: (criteria string, point_weighting)}
        
        

        Parameters
        ----------
        df : DataFrame
            The curve data DataFrame that the new column should be created in.
        criteria_dict : dict
            A dictionary of criteria to flag rows as possible pay. Eg.
            {'RESD': ('> 200', 5), "IC5_GAS_LOG10": ('> 3', 2)}
        formation_pen_tuple : tuple
            A tuple that preserves the model scores in the specified formation,
            otherwise a penalty is applied in order to show better results in
            the formation where the screening model likely applies.
            eg. ('Gacheta', 0.5)
        

        Returns
        -------
        Original DataFrame with new column added

        """
        
        strat_unit_name = 'Strat_unit_name'
        
        num_params = len(criteria_dict)
        score_weights = [i[1] for i in criteria_dict.values()]
        total_weights = sum(score_weights)
        
        
        # Initialize the score column with zeros
        df['Gradational_flag'] = float(0)
        num_conditions = len(criteria_dict)
        
        # Initialize a mask for all True values
        mask = pd.Series([True] * len(df), index=df.index)
        
        # Iterate through the conditions and apply them sequentially
        for i, (curve_name, (curve_condition, curve_weight)) in enumerate(criteria_dict.items()):
            print(i, curve_name, curve_condition, curve_weight)
            operator, value = curve_condition.split(' ')
            try: 
                value = float(value)  # Convert value to float for comparison
            except ValueError:
                value = value.strip("'\"") #Remove any surrounding quotes from string
            # Apply the condition to the entire DataFrame
            if operator == '>':
                current_mask = df[curve_name] > value
            elif operator == '>=':
                current_mask = df[curve_name] >= value
            elif operator == '<':
                current_mask = df[curve_name] < value
            elif operator == '<=':
                current_mask = df[curve_name] <= value
            elif operator == '==':
                current_mask = df[curve_name] == value
            elif operator == '!=':
                current_mask = df[curve_name] != value
            else:
                raise ValueError(f"Unsupported operator: {operator}")
                
            print('Current mask', sum(current_mask))
            
            # Assign points to rows that satisfy the current mask
            df.loc[current_mask, 'Gradational_flag'] += (curve_weight / total_weights)
            print (curve_weight/total_weights)
        
        #If a formation penalty tuple is passed, apply the formation penalty
        #to rows where the formation is not matched
        if formation_pen_tup is not None:
            formation, penalty_factor = formation_pen_tup
            penalty_mask = df[strat_unit_name] != formation
            
            df.loc[penalty_mask, 'Gradational_flag'] *= penalty_factor
            
                
        return df

class CleanLogData:
    def __init__(self):
        pass
    curve_limit_dictionary = {'BS': (0,30), #Borehole size 0 to 30 inches
                         'CALI': (0,30), #Caliper 0 to 30 inches
                         'DRHO': (-1, 1.5), #Limits for density correction
                         'DT': (30, 300),
                         'GR': (0,300),
                         'DTS': (50,900),
                         'RHOB': (1,4),
                         'NPHI': (-0.15,1.2),
                         'NPSS': (-0.15, 1.2),
                         'NPLS': (-0.15,1.2),
                         'NPDL': (-0.15,1.2),
                         'PEF': (0,15),
                         'RESS': (0.001,100000),
                         'RESM': (0.001,100000),
                         'RESD': (0.001,100000),
                         'RESS_LOG10': (-3,5),
                         'RESM_LOG10': (-3,5),
                         'RESD_LOG10': (-3,5),
                         'C1_GAS': (0,1000000),
                         'C2_GAS': (0,1000000),
                         'C3_GAS': (0,1000000),
                         'NC4_GAS': (0,1000000),
                         'IC5_GAS': (0,1000000),
                         'NC5_GAS': (0,1000000),
                         'TOTALGAS': (0,1000000),
                         'C1_GAS_LOG10': (0,6),
                         'C2_GAS_LOG10': (0,6),
                         'C3_GAS_LOG10': (0,6),
                         'IC4_GAS_LOG10': (0,6),
                         'NC4_GAS_LOG10': (0,6),
                         'IC5_GAS_LOG10': (0,6),
                         'NC5_GAS_LOG10': (0,6),
                         'SP': (-500,500),
                         'ROP': (0,2000),
                         'TEMP': (0,500),
                         'X_COORD': (600000, 2000000),
                         'Y_COORD': (400000, 2000000)
                         }
    
    columns_for_analysis = ['GR',
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
                            'ROP'
                            ]
        
    def mixed_type_aggregation(df, numeric_agg_type='mean'):
        all_columns = list(df.columns)
        numeric_columns = list(df.select_dtypes(include=[np.number]).columns)
        non_numeric_columns = [column for column in all_columns if column not in numeric_columns]
        
        aggregation_dict = {x: numeric_agg_type for x in numeric_columns}
        for y in non_numeric_columns:
            aggregation_dict[y] = 'first'
        
        return aggregation_dict

    def apply_curve_limits(df, curve_limits_dict, drop_na=True):
        curve_columns = list(df.columns)
        
        curves_in_dict = list(curve_limits_dict.keys())
        
        curves_to_apply = [curve for curve in curve_columns if curve in curves_in_dict]
        
        if drop_na == True:
            
            for curve in curves_to_apply:
            
                size_before = df.shape[0]
                
                curve_lim_low, curve_lim_high = curve_limits_dict[curve]
                
                df = df[((df[curve] >= curve_lim_low) & (df[curve] <= curve_lim_high)) | (df[curve].isna())]
                
                size_after = df.shape[0]
                
                difference = size_before - size_after
                
                print(f'For curve {curve}, {difference} rows were eliminated')
        
        else:
            for curve in curves_to_apply:
                
                curve_lim_low, curve_lim_high = curve_limits_dict[curve]
                
                mask = (df[curve] <= curve_lim_low) | (df[curve] >= curve_lim_high)
                
                mask_len = len(mask)
                print(curve)
                print(f'Mask length = {mask_len}')
                
                mask_sum = sum(mask)
                print(f'Mask Sum = {mask_sum}')
                
                df[curve][mask] = np.nan #Mask has invalid values, turn these to nan
            
        return df
    
    #Clean up NPHI, which is sometimes expressed in porosity units (e.g. 15 pu) instead of a fraction (e.g. 0.15)
    def NPHI_to_fraction(mega_df, aggregation_dictionary):
        
        NPHI_prescreen_limit_lower = -15
        NPHI_prescreen_limit_upper = 120
        

        # Function to replace values with NaN
        def replace_with_nan(x):
            if NPHI_prescreen_limit_lower <= x <= NPHI_prescreen_limit_upper:
                return x
            else:
                return np.nan
        
        # Apply function to the column
        mega_df['NPHI'] = mega_df['NPHI'].apply(replace_with_nan)
        
        #Group by well name
        gb1 = mega_df.groupby(level=0)
        
        #Aggregate according to aggregation dict. This is helpful to handle non-numeric values
        gb_df = gb1.agg(aggregation_dictionary)
        
        #Threshold, over this value it is relatively certain that values are being expressed in porosity units
        NPHI_threshold = 6
        
        NPHI_wells_to_correct_df = gb_df[gb_df['NPHI'] > NPHI_threshold]
        
        #Make a list of unique wells that appear to have NPHI expressed in porosity units
        NPHI_wells_to_correct = list(NPHI_wells_to_correct_df.index.get_level_values(0).unique())
        
        #Divide the NPHI column for the identified wells by 100 (conversion from PU to fraction)
        mega_df.loc[mega_df.index.get_level_values(0).isin(NPHI_wells_to_correct), 'NPHI'] /= 100
        
        return mega_df
    
    
        
def plot_attribute_map_with_df(filtered_df, 
                       column,
                       x_lims=None,
                       y_lims=None,
                       log_normalized_column=None,
                       size_clip=None,
                       color_lims=None,
                       save_img=False,
                       plot_title=None,
                       plot_xlabel=None,
                       plot_ylabel=None,
                       plot_legend_label=None,
                       scale_factor=10,
                       fig_size_x=30
                       ):
    """
    This is a function that plots an attribute scatter map with a block map
    as a background based on well object dataframe data. The data must be
    preconditioned before entering this function

    Parameters
    ----------
    filtered_df : DataFrame
        A DataFrame with all stratigraphic and petrophysical prefiltering applied.
    column : str
        The column corresponding to the target data column to plot.
    x_lims : Tuple, optional
        A tuple of the desired x limits for the map. The default is None.
    y_lims : Tuple, optional
        A tuple of the desired y limits for the map. The default is None.
    log_normalized_column : str, optional
        The column with log normalized values of the target column. Better to
        use for displaying size than raw log scale values
    size_clip : Tuple, optional
        A tuple of values at which to clip the data to show better variability
    color_lims : Tuple, optional
        A tuple of max-min values over which to normalize colors
    save_img : Boolean, optional
        Choose whether a png image of the plot will be saved. The default is False.
    plot_title : str, optional
        The title to assign to the plot. The default is None.
    plot_xlabel : str, optional
        The x axis label for the plot. The default is None.
    plot_ylabel : str, optional
        The y axis label for the plot. The default is None.
    plot_legend_label : list, optional
        A list showing labels for the plot legend. The default is None.
    scale_factor : int, optional
        The factor by which to multiply the size of the points. The default is 5.
    fig_size_x : int, optional
        The size of the plot on the x axis. The y axis size is scaled
        automatically in accordance with the map ratio and the x axis size.
        The default is 20.

    Returns
    -------
    None.

    """
    #%%
    x_coord_column = 'X_COORD'
    y_coord_column = 'Y_COORD'
    
    # Filter out only relevant columns
    columns_for_plotting = [x_coord_column, y_coord_column, column]
    
       
    filtered_df = filtered_df[columns_for_plotting]
    
    #Create points for wells
    well_points_geometry = [Point(xy) for xy in zip(filtered_df['X_COORD'], filtered_df['Y_COORD'])]
    well_points_geo_df = gpd.GeoDataFrame(filtered_df, geometry=well_points_geometry)
    
    #%%
    #Read shapefile
    shapefile_fn = r"C:\Programming\AI_Production_Log_Project_2023\Scripting Session Folders\26MAY2024 GeoPandas Session\mapping_log_data\Tierras_Marzo_Shape_290323_EPSG3116.shp"
    tierras_geodataframe = gpd.read_file(shapefile_fn)
    
    #Flag Parex blocks
    parex_mask = tierras_geodataframe['OPR_ABR'] == 'PAREX'
    
    #Create a mask with Parex blocks
    tierras_geodataframe['PAREX_OPERATOR'] = parex_mask
    
    #%%
    #Establish map limits
    if x_lims == None:
        x_min = filtered_df['X_COORD'].min()
        x_max = filtered_df['X_COORD'].max()
    else:
        x_min = x_lims[0]
        x_max = x_lims[1]
    
    if y_lims == None:
        y_min = filtered_df['Y_COORD'].min()
        y_max = filtered_df['Y_COORD'].max()
    else:
        y_min = y_lims[0]
        y_max = y_lims[1]
        
    y_x_ratio = (y_max - y_min) / (x_max - x_min)
    
    
    fig_size_y = fig_size_x * y_x_ratio
    
    fig_size = (fig_size_x, fig_size_y)    
        
    # ax = tierras_geodataframe.plot(cmap='jet', edgecolor='black',figsize=(30,30), column='PAREX_OPERATOR', alpha=0.5)
    ax = tierras_geodataframe.plot(cmap='jet', edgecolor='black', figsize=fig_size, column='PAREX_OPERATOR', alpha=0.2)
   
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    if plot_legend_label == None:
        plt.legend(['Non-Parex', 'Parex'])
    else:
        plt.legend(plot_legend_label)
#%%



    #If there is a log normalized version of the target column specified,
    #use this to specify the size
    if log_normalized_column is not None:
        size_raw = filtered_df[log_normalized_column]
        #Clip data, if necessary, to better show the variability
        if size_clip != None:
            lower_sz, upper_sz = size_clip
            size_raw = size_raw.clip(lower_sz, upper_sz)
        
        
        size_raw_min = size_raw.min()
        #Set minimum value for size to 1
        size = size_raw - size_raw_min + 1
        
        #If there are color limits to apply, do it here
        if color_lims != None:
            color_min, color_max = color_lims
            colour_normalization = mcolors.LogNorm(vmin=color_min, vmax=color_max)
        #Otherwise, normalize colours on min and max values
        else:
            color_min = filtered_df[column].min()
            color_max = filtered_df[column].max()
            
            colour_normalization = mcolors.LogNorm(vmin=color_min, vmax=color_max)
    
    #If the data is not log normalized, normalize it linearly and manually
    else:
        size = filtered_df[column]
        
        color_min = filtered_df[column].min()
        color_max = filtered_df[column].max()
        
        colour_normalization = mcolors.Normalize(vmin=color_min, vmax=color_max)
    

    size = size * scale_factor
    colour = filtered_df[column]
    
    ## size = filtered_df[column] * scale_factor
    ## colour = filtered_df[column]
    
    
    
    ## colour_normalization = mcolors.Normalize(vmin=-2, vmax=2)
    
    
    
    ax = well_points_geo_df.plot(ax=ax, 
                            kind='scatter', 
                            x='X_COORD', 
                            y='Y_COORD', 
                            s=size, 
                            c=colour, 
                            cmap='viridis', 
                            norm=colour_normalization)
    
    
    #%%
    
    if plot_title is not None:
       plt.title(plot_title, fontsize=30)
    else:
       plt.title(f'Map of {column}')
       
    if plot_xlabel is not None:
       plt.xlabel(plot_xlabel, fontsize=15)
    else:
       plt.xlabel('X', fontsize=15)
       
    if plot_ylabel is not None:
       plt.ylabel(plot_ylabel, fontsize=15)
    else:
       plt.ylabel('Y', fontsize=15)


    if (save_img == True) and (plot_title != None):
        plt.savefig(f'{plot_title}.png')

    return ax

class LogDataFilter:
    """
    This class is used to filter log data from a mega dataframe.
    """
    def __init__(self):
        pass
    def well_name_filter_regex(well_list):
        """
        Construct the regex to filter a dataframe based on a list of well names
        
        Parameters
        ----------
        well_list : list
            List of well UWIs

        Returns
        -------
        well_string_regex : str
            The regex expression corresponding to the list of wells.

        """
        print('Well list length' + str(len(well_list)))
        
        if len(well_list) == 1:
            result = well_list[0]
            
            return result
        else:
            well_string_regex = well_list[0]
            for well in well_list[:-1]:
                well_string_regex = well_string_regex + '|' + well
            well_string_regex = well_string_regex + well
        return well_string_regex
    
    
    def well_name_filter(df, well_list, well_index_name='Well_name', exact_match=False):
        """
        Filter a dataframe based on well names, or field
        eg. 'KANA' in well list will return all the wells from the Kananaskis field
        'KANA0002', 'KANA0003' will return the Kananaskis-2 and Kananaskis-3 wells

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        well_list : TYPE
            DESCRIPTION.
        well_index_name : TYPE, optional
            DESCRIPTION. The default is 'Well_name'.

        Returns
        -------
        filtered_df : TYPE
            DESCRIPTION.

        """
        if exact_match == True:
            df_list = []
            for well in well_list:
                filtered_df = df[df.index.get_level_values(well_index_name) == well]
                df_list.append(filtered_df)
            
            concat_df = pd.concat(df_list)
            return concat_df
        
        elif exact_match == False:
            well_string_regex = LogDataFilter.well_name_filter_regex(well_list)
            filtered_df = df[df.index.get_level_values('Well_name').str.contains(well_string_regex)]
            
            return filtered_df
        
    def curve_data_filter(df, curve_filter_dict):
        """
        Filter a DataFrame based on certain specified conditions
        
        Parameters
        ----------
        df : DataFrame
            Source dataframe from which to take slice..
        curve_filter_dict : dict
            Dictionary of conditions with following structure {curve_name: (operator value)} all as strings
            eg. {'GR': '> 30'}

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        df_filtered : DataFrame
            DataFrame filtered with specified conditions.

        """
        mask = pd.Series([True] * len(df), index=df.index)
        for curve, condition in curve_filter_dict.items():
            operator, value = condition.split(' ')
            try:
                value = int(value)
            except ValueError:
                value = float(value)
                
            if operator == '>':
                mask &= df[curve] > value
            elif operator == '>=':
                mask &= df[curve] >= value
            elif operator == '<':
                mask &= df[curve] < value
            elif operator == '<=':
                mask &= df[curve] <= value
            elif operator == '==':
                mask &= df[curve] == value
            elif operator == '!=':
                mask &= df[curve] != value
            else:
                raise ValueError(f"Unsupported operator: {operator}")
                
        df_filtered = df[mask]
        
        return df_filtered
    
    def formation_filter(df, formation_name_list):
        if len(formation_name_list) == 1:
            formation_name = formation_name_list[0]
            result = df[df['Strat_unit_name'] == formation_name]
            
            return result
        else:
            df_list = []
            for formation_name in formation_name_list:

                df_temp = df[df['Strat_unit_name'] == formation_name]
                df_list.append(df_temp)
            result = pd.concat(df_list)
        return result
    
    def depth_slice(df, well_depth_dict):
        """
        

        Parameters
        ----------
        df : DataFrame
            Source dataframe from which to take slice.
        well_depth_dict : dict
            Dictionary of well slices with {well_name: (top_of_interval, bottom_of_interval)} structure.
            The slices can either be passed as one list or tuple, or 

        Returns
        -------
        final_df : DataFrame
            Concatenated DataFrame with well slices.

        """
        
        dfs = []
        for well_name, depth_slice in well_depth_dict.items():
            first_slice_element = depth_slice[0]
            if not isinstance(first_slice_element, (list,tuple)):
                top_of_interval, base_of_interval = depth_slice
            
                df_temp = df.loc[(well_name, top_of_interval):(well_name, base_of_interval)]
                dfs.append(df_temp)
            else:
                depth_slices = depth_slice
                for depth_slice in depth_slices:
                    top_of_interval, base_of_interval = depth_slice
                
               
                    df_temp = df.loc[(well_name, top_of_interval):(well_name, base_of_interval)]
                    dfs.append(df_temp)
                    
        
        final_df = pd.concat(dfs)
        
        return final_df
    
    def get_well_names_from_df(df, uwi_index='Well_name'):
        
        #Get all entries in the Well_name index of the multiindex
        well_name_index = df.index.get_level_values('Well_name')
        
        #Remove duplicates and turn into list
        well_name_list = list(set(list(well_name_index)))
        
        #Sort alphabetically
        well_name_list.sort()
        
        return well_name_list
    
    
    
    lithology_curves = ['GR',
                        'RHOB',
                        'NPHI',
                        'PEF',
                        'DT',
                        'SP']
        
        
        
                   

# def create_super_df_(ascii_data_list):
#     super_list = ascii_data_list
#     #super_list = [i.A for i in logs.values()] #Use list comprehension to extract ASCII data
#     super_df = super_list[0] #Initiate dataframe with first entry from dataframe list
#     #for each subsequent dataframe, merge on the DEPT column
#     for i in range(len(super_list))[1:]:
#         print(i)
#         super_df = pd.concat([super_df, super_list[i]], axis=0, ignore_index=True) #Concatenate ASCII data in list to dataframe
    
#     super_df = pd.concat(super_list)
#     return super_df

# if __name__ == "__main__":
    
#     logs = create_logs(logs_dir) #Add
    
#     super_df = create_super_df(logs)
    
    
    
    
    
    
    
    
    
    #conditions = "Fmn_top == 'Mirador', ResD > 200"
    #conditions = "Fmn_top == 'Gacheta', GR > 50"
    
    #Parse the string that lays out the conditions for the fancy index into
    #a list that can be understood by the filter_data() function
    
    #filtered_data = filter_data(super_df, conditions)
    





    
#    plot = crossplot_data(filtered_data, 'delta_ND', 'ResD')
    
    
                               
#    counted_data = count_filtered_data(filtered_data, logs)
#    
#    ax1 = create_data_map(counted_data, logs, map_filename)

#vert_bar_chart(counted_data)




#df1 = logs['Tilo_1'].A
#df2 = logs['Bacano_3'].A
#df3 = logs['Corcel_A1'].A
#       
    
#L2 = []
#
#for log in d1.values():
#    log_data = log.A
#    log_data['Well_Name'] = log.W.UWI.value
#    if 'GR' and 'ResD' and 'Fmn_top' in log_data.columns:
#        filter_data = log_data[(log_data['GR'] < 50) & (log_data['ResD'] > 500) & (log_data['Fmn_top'] == 'Mirador')]
#        filter_data['concat_index'] = filter_data.index.map(str) + filter_data['Well_Name']#('{}_{}').format(filter_data.index(), filter_data['Well_Name'])
#        filter_data.reset_index().set_index('concat_index')
#        L2.append(filter_data)
#
#if len(L2) > 0:
#    df_results = L2[0]
#    for i in range(1,len(L2)):
#        df_results += L2[i] 

        
    