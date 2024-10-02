# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:58:22 2024

@author: Gfoo
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

from . import plastic as pl
from . import meta_plastic
from . import curve_plotting
from . import merge_multiple_las_gas_electric_logs as merge



if __name__ == "__main__":
    
    mega_log_dict_fn = r"C:/Programming/AI_Production_Log_Project_2023/Scripting Session Folders/22APR2024 Bypassed Pay Session/Logs/Mega Log Dictionary/No tops/mega_log_dict_22APR2024.pkl"
    
    mega_log_dict = pd.read_pickle(mega_log_dict_fn)
        
    import manage_tops

    tops_df_filename = r"C:\Programming\AI_Production_Log_Project_2023\Scripting Session Folders\22APR2024 Bypassed Pay Session\Tops\Bypassed_Pay_AI_Tops_18APR2024_replaced.csv"

    permitted_tops_and_bases_filename = r"C:\Programming\AI_Production_Log_Project_2023\Scripting Session Folders\22APR2024 Bypassed Pay Session\Tops\kingdom_lithostrat_tops_and_bases 17APR2024.csv"

    interpreter_preference_list = ['GAF', 'GAF2', 'PXT', 'XRA', 'DGH', 'RGH', 'MWB']   
        
    manage_tops.apply_formation_tops_well_log_dict(mega_log_dict, tops_df_filename, permitted_tops_and_bases_filename, interpreter_preference_list)
    
    meta_plastic.apply_XY_coords_to_dict(mega_log_dict)
    
    mega_df = meta_plastic.create_super_df(mega_log_dict)
               