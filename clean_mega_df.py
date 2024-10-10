# -*- coding: utf-8 -*-
"""
Created on Fri May 31 13:08:43 2024

@author: Gfoo
"""
import pandas as pd
import numpy as np



#%%
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
            
            
            df[mask] = np.nan
        
    return df
        

curve_limits_dictionary = {'BS': (0,30), #Borehole size 0 to 30 inches
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

key_curves_list = ['DT', 
                   'GR', 
                   'NPHI', 
                   'PEF', 
                   'RHOB', 
                   'SP', 
                   'Strat_unit_name',
                   'DTS',
                   'ROP', 
                   'NPSS',
                   'OIL_SHOW', 
                   'TOTALGAS', 
                   'TEMP',
                   'RESD_LOG10',
                   'RESM_LOG10', 
                   'RESS_LOG10', 
                   'C1_GAS_LOG10',
                   'C2_GAS_LOG10', 
                   'C3_GAS_LOG10',
                   'IC4_GAS_LOG10',
                   'NC4_GAS_LOG10',
                   'NC5_GAS_LOG10',
                   'IC5_GAS_LOG10',
                   ]
         
      


#%%
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


def mega_df_clean_routine(mega_df):
    #Add LOG10 curves
    candidate_curves_to_convert = ['RESD','RESM','RESS','C2_GAS','C3_GAS','C1_GAS','IC5_GAS',
           'NC4_GAS', 'NC5_GAS', 'TOTALGAS']
    
    curves_to_convert = [curve for curve in candidate_curves_to_convert if curve in mega_df.columns]
    
    for curve in curves_to_convert:
        log10_curve_name = f'{curve}_LOG10'
        mega_df[log10_curve_name] = mega_df[curve].apply(np.log10)
            
        #Get object for negative infinity
        neg_inf = np.log10(0)
        
        #Replace negative infinity with np.nan
        mega_df = mega_df.replace(neg_inf, np.nan)


    aggregation_dict = mixed_type_aggregation(mega_df, 'median')

    mega_df = NPHI_to_fraction(mega_df, aggregation_dict)
    
    return mega_df


#%%

if __name__ == "__main__":
    mega_df = pd.read_pickle(r"C:\Programming\AI_Production_Log_Project_2023\Scripting Session Folders\22APR2024 Bypassed Pay Session\Logs\Mega Dataframe\With tops and coordinates and log10 curves\mega_df_with_tops_and_coordinates_22MAY2024.pkl")

    aggregation_dict = mixed_type_aggregation(mega_df, 'median')