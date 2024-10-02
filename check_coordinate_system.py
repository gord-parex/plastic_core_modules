# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:14:52 2024

@author: Gfoo
"""

import numpy as np
import pandas as pd
from . import plastic as pl
from . import meta_plastic
from . import test_classification_workflow as tcw
from . import well_name_translation as well_translate
from . import merge_multiple_las_gas_electric_logs as merge_logs
import pickle as pkl
from . import histogram_plotting
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pyproj import Transformer, CRS

 


coord_system_modification_dict = {'MAGNA COLOMBIA ESTE': 'EPSG:3117',
                                  'MAGNA COLOMBIA ESTE ESTE': 'EPSG:3118',
                                  'MAGNA COLOMBIA OESTE': 'EPSG:3115',
                                  'COLOMBIA E CENTRAL ZONE': 'EPSG:21898',
                                  'COLOMBIA BOGOTA ZONE': 'EPSG:21897',
                                  'COLOMBIA WEST ZONE': 'EPSG:21896',
                                  }


def modify_coord_system_from_WGS84(row):
    wgs84_system = CRS('EPSG:4326')
    magna_sirgas_origen_bogota_system = CRS('EPSG:3116')
    
    latitude = row['WELL_LATIT']
    longitude = row['WELL_LONGI']
    
    try:
        assert isinstance(latitude, (float, int))
        assert isinstance(longitude, (float, int))
    except AssertionError:
        print('Input type was invalid')
        
    transformer = Transformer.from_crs(wgs84_system, magna_sirgas_origen_bogota_system, always_xy=True)
    
    xcoord_magna_sirgas, ycoord_magna_sirgas = transformer.transform(longitude, latitude)
    
    row['XCOORD_MAGNA_SIRGAS_BOGOTA_PXT'] = xcoord_magna_sirgas
    row['YCOORD_MAGNA_SIRGAS_BOGOTA_PXT'] = ycoord_magna_sirgas
    
    return row
    
    
    

def modify_coord_system_from_xy(row):
    target_column = 'COORD_SYSTEM'
    coord_system_name = row[target_column]
    if type(coord_system_name) != np.ndarray:
        if coord_system_name in coord_system_modification_dict.keys():
            print(f'Target label {coord_system_name} found')
            epsg_original = coord_system_modification_dict[coord_system_name]
            epsg_target = 'EPSG:3116'
            
            trans = Transformer.from_crs(
                epsg_original,
                epsg_target,
                always_xy=True,
                )
            
            xx, yy = trans.transform(row['X_COORD'],
                                     row['Y_COORD']
                                     )
            row['OLD_XCOORD'] = row['X_COORD']
            row['OLD_YCOORD'] = row['Y_COORD']
            
            row['X_COORD'] = xx
            row['Y_COORD'] = yy
            row['COORD_MODIFIED'] = 1
            # print(f'Modified')
            # print(row)
    
    return row
                             
    
#%%
if __name__ == "__main__":
    mega_df_fn = r"C:\Programming\AI_Production_Log_Project_2023\Scripting Session Folders\22APR2024 Bypassed Pay Session\Logs\Mega Dataframe\With tops and coordinates and log10 curves\PRE_TRANSFORMATION COPY\mega_df_with_tops_and_coordinates_pre_transformation_22MAY2024.pkl"
    mega_df = pd.read_pickle(mega_df_fn)
    

#%%
    
    #Group by well name
    gb1 = mega_df.groupby(level='Well_name')
    
    #Take mean of numeric columns, to get a df with only 1 row per well
    df_gb1 = gb1.mean(numeric_only=True)
    
    #Use this aggregation method to return the most common label in the dataframe 
    #for the coodinate system column when, by well
    df_gb1['COORD_SYSTEM'] = gb1['COORD_SYSTEM'].agg(lambda x: x.mode())
    
    #Reset index to get Well_name as a column
    df_gb1 = df_gb1.reset_index()
    
    
    
    #%%
    #Get a dataframe with coordinate data only
    df_gb1_coords_only = df_gb1[['Well_name', 'X_COORD', 'Y_COORD', 'COORD_SYSTEM']]
    
    #%%
    #Add a column 'COORD_MODIFIED' to show whether the coordinates were transformed or not, instantiate at 0
    df_gb1_coords_only['COORD_MODIFIED'] = 0
    
    
    #%%
    #Run the modify_coord_system() function on coordinates only dataframe
    df_gb1_coords_only = df_gb1_coords_only.apply(modify_coord_system, axis=1)
    
    print('Coord system change complete')
    
    #%%
    #Drop X_COORD and Y_COORD columns in preparation for merging of dataframes
    mega_df1 = mega_df.drop(columns=['X_COORD', 'Y_COORD'])
    
    
    #%%
    #Merging dataframes is the easiest way to get the information from the 
    #one-row-for-one-well coordinates-only dataframe into the mega_dataframe
    mega_df1 = mega_df1.reset_index().merge(df_gb1_coords_only, on='Well_name').set_index(['Well_name', 'DEPT'])
    
    
    #%%
    
    #Create a column that is simply 'COORD_SYSTEM'
    mega_df1['COORD_SYSTEM'] = mega_df1['COORD_SYSTEM_x']
    
    #Drop redundant 'COORD_SYSTEM_x' and 'COORD_SYSTEM_y' columns (which are
    #artefacts of the df merge) in mega_df1 dataframe
    mega_df1 = mega_df1.drop(columns=['COORD_SYSTEM_x', 'COORD_SYSTEM_y'])
    
    #Drop the additional coord system data for mega_df3, which leaves a 'clean'
    #dataframe to export
    mega_df3 = mega_df1.drop(columns=['OLD_XCOORD', 'OLD_YCOORD', 'COORD_MODIFIED', 'COORD_SYSTEM'])








    
    


    
    