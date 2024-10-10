# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 13:55:57 2024

@author: gfoo
"""


"""
Models to screen for pay. To be used in conjunction with 
meta_plastic.create_gradational_flag_curve()

"""


class GachetaModels():
    def __init__(self):
        pass
                    
    kananaskis_gacheta_gas_pay = {'RESD': ('> 50', 3), 
                                  'NC5_GAS_LOG10': ('> 1.5', 2),
                                  'TOTALGAS_LOG10': ('> 3', 5),
                                  'OIL_SHOW_BINARY': ('== 1', 1),
                                  'Haworth_fluid_classification': ("== 'Gas'", 2),
                                  'DELTA_PHID_PHIN': ('< 0', 4),
                                  }
    
    kananaskis_gacheta_oil_pay = {'RESD': ('> 50', 5), 
                                  'NC5_GAS_LOG10': ('> 1.5', 2),
                                  'TOTALGAS_LOG10': ('> 3', 5),
                                  'OIL_SHOW_BINARY': ('== 1', 1),
                                  'Haworth_fluid_classification': ("== 'Oil'", 2)
                                  }
    

    las_maracas_gacheta_oil_pay = {'RESD': ('> 10', 5),
                                   'RESD': ('> 30', 5),
                                  'DELTA_PHID_PHIN': ('< 0.23', 5),
                                  'WETNESS_RATIO': ('> 25', 2),
                                  'WETNESS_RATIO': ('> 50', 2),
                                  'OIL_SHOW_BINARY': ('> 0', 3),
                                  'C3_GAS_LOG10': ('> 1', 2),
                                  'NC4_GAS_LOG10': ('> 1', 2),
                                  'NC5_GAS_LOG10': ('> 1', 2),
                                  'NC5_GAS_LOG10': ('> 1.6', 2),
                                   }


