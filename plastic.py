# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 00:44:09 2017

LAST UPDATED 27MAY2024

@author: Gordon Foo
All rights reserved
"""

import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
import pdb

#TODO:
#Error handling is non-existent
#None of this code has been optimized for speed


def create_log(las_filename, well_name = None, tops_filename = 0):
       
    new_log = Log(las_filename, well_name, tops_filename)
    return new_log


class Log(object):
    """Creation of Log object. Log object stores many of the common attributes defined in the CWLS standard for LAS files
    """
    
    def __init__(self, las_FN, well_name, tops_FN=0):
        #las_FN is the filepath for the LAS data, tops_fn is the filepath for the formation tops to assign to the log object. tops_FN is an optional argument
        self.name = well_name
        #keep the raw data dictionary with the lines for each section
        #stored as an attribute of the log class
        self.raw_data_dict = self.split_sections(las_FN)
        
        
        if tops_FN != 0:
            self.tops_filename = tops_FN
        else:
            self.tops_filename = None
        self.assign_sections()
        
        self.assign_curve_data()
        
        self.curve_limits = self.get_curve_limits()
        
        #If no well name is specified, see if the attribute self.W.WELL is an
        #object of the attribute class. If so, assign the value of the 'WELL'
        #attribute to be the 'name' attribute of the log object. This gives
        #the user easier access to the full-length name of the well, not just
        #the UWI. The UWI is usually a modified name rather than the full well
        #name
        if well_name == None:
            if str(type(self.W.WELL)) == "<class 'plastic.Well_Attribute'>":
                #Set name attribute
                self.name = self.W.WELL.value
            else:                
                #Split on slash, take the last value, then split on period and
                #take the second-last value (the last value after this split
                #would be the file extension, '.las')
                self.name = (las_FN.split('/')[-1]).split('.')[-2]   
                

        
            
    def read_las(self, las_FN):
        """Read the LAS file to a list called linedata, which separates all of the lines in the LAS file to separate entries in a list"""
        with open(las_FN) as f:
            linedata = f.read().splitlines()
            f.close()
    
        #Remove all lines that begin with '#' to avoid errors in execution
        linedata = [i for i in linedata if len(i) > 0]
        linedata = [i.strip() for i in linedata if i.strip()[0] != '#']
        
        return linedata

    def get_section_flags(self, linedata):
        """Look for section flage in the linedata list. This lets the interpreter know how many "section" objects to create and with what name"""
        #Make list of flags by searching for tilde (~) as the first char in line        
        section_flags = [i[:2] for i in linedata if i[0] == '~']
        #Check if version and ASCII data flags in right place, per LAS 2.0 standard
        assert section_flags[0] == '~V' and section_flags[-1] == '~A'        
        
        return section_flags
    
    def get_section_indicies(self, linedata):
        #Return positions of lines that begin with ~ (This is a list of the 
        #location where each section begins)
        section_indicies = [i for i,j in enumerate(linedata) if linedata[i][0] == '~']
        return section_indicies


    def split_sections(self, las_FN, do_print=False):
        linedata = self.read_las(las_FN)
        
        raw_data_dict = {}
        
        section_flags = self.get_section_flags(linedata)
        section_indicies = self.get_section_indicies(linedata)
        #print('Section flags are ', section_flags)
        #print('Section indicies are ', section_indicies)

        
        for i in range(len(section_flags)):
            #Index of start of slice is value stored in the position i of
            #sect_indicies_list. Add one so it doesn't inlude line with flag
            start = section_indicies[i] + 1
            #If section is ASCII data, slice to end (stop=None), othersise, slice
            #to the start of the next section (stop = section_indicies[i+1])
            if section_flags[i] == '~A':
                stop = None
            else:
                stop = section_indicies[i+1]             
            #Get the flag in position i
            section_flag = section_flags[i]
            #Get a slice corresponding to the start and stop indicies
            section_data = linedata[start:stop]
            
            #Update dictionary with flag as key, slice of data as value
            raw_data_dict.update([(section_flag, section_data)])
            
            if do_print == True: 
                print('raw_data_dict keys are ', raw_data_dict.keys())
        #print('raw_data_dict is ', raw_data_dict)   
        return raw_data_dict
    


        
    def assign_sections(self):
        raw_data_dict = self.raw_data_dict
        #Here we want to return a dictionary of Well_Section objects. Take all the data for that
        #section and run it through raw section then populate a dictionary which will
        
        #Instantiate section_dict
        section_dict={}
        
        #Put key of section dict as second character of key of raw_data_dict
        parsable_sections = ['~V','~W','~C','~P']
        
        #Loop over section names in raw_data_dict and parse the section if it
        #is a parsable section. Otherwise, directly assign the data to the
        #attribute
        for raw_section_name, data in raw_data_dict.items():
            if raw_section_name in parsable_sections:
                cleaned_section_name = raw_section_name[-1]
                section_dict[cleaned_section_name] =  self.parse_section(data)
            
            #Generally, this else condition should only apply to the '~O' or
            #'other' data section from the LAS file
            else:
                setattr(self, raw_section_name[-1], data)

        for section_name, section_data in section_dict.items():
            setattr(self, section_name, section_data)
    
    def parse_section(self, section_data):
        #Section data is raw data for a particular setion, this is a list of strings
        
        #Instantiate ordered dict
        section_dict = OrderedDict()
        #print (section_data)
        for i,j in enumerate(section_data):
            #print ('i is {}').format(i)
            #print ('section_data[i] is {}').format(section_data[i])
            try:
                if section_data[i][0] != '#':
                    attribute = self.split_attribute(section_data[i])
                    #print('Attribute is', attribute)
                else:
                    continue
            except IndexError:
                attribute = Well_Attribute('Invalid', 'Invalid', 'Invalid', 'Invalid')
            #The mnemonic, which will become the key for the attribute in
            #the section dictionary, is accessed as the .mnem attribute of the 
            #Well_Attribute object that was returned by self.split_attribute()
            mnem = attribute.mnem
            #print('mnem is ', mnem)
            
            section_dict.update({mnem: attribute})
#            print ('Section dict is {}').format(section_dict.values)

        #Return a Well_Section object that is called with the section_dict
        #ordered dictionary. Calling this will automatically assign all of the
        #attribute entries in the section_dict dictionary to attributes of the 
        #Well_Section object
        
        return Well_Section(section_dict)  
    
    def split_attribute(self, line):
        #Take the string, which is the raw data for the well attribute, and 
        #split it sequentially, first with period, then space, then colon.
        #Pass these 4 entries to the Well_Attribute class to be made into a
        #well attrubute object
        a1 = line.split('.',1)[0].strip()
        #print('a1 is ', a1)
        a2 = line.split('.',1)[1].split(' ',1)[0].strip()
        #print('a2 is ', a2)
        a3 = line.split('.',1)[1].split(' ',1)[1].split(':',1)[0].strip()
        #print('a3 is ', a3)
        a4 = line.split('.',1)[1].split(' ',1)[1].split(':',1)[1].strip()
        #print('a4 is ', a4)
        #strip out whitespace at beginning and end with strip() method
        attribute = Well_Attribute(a1, a2, a3,a4) 
       
        return attribute

        
    def assign_curve_data(self):
        #From the raw sections data (not yet digested into a section object),
        #take out the lines associated with the key '~A', which  is the ASCII
        #curve data
        raw_curve_data = self.raw_data_dict['~A']
        
        
        #The names of the curves are an attribute of the section object C, 
        #which is an attribute of the Log object
        curve_names = self.C.mnems
        
        uwi = str(self.W.UWI.value)
        
        if uwi is not None:
            well_name = str(self.W.UWI.value)
        else:
            well_name = str(self.name)

        setattr(self, 'curve_names', curve_names)
        
        null_value = float(self.W.NULL.value)
        
        
        #pdb.set_trace()        
        
        df1 = pd.DataFrame([row.split() for row in raw_curve_data], columns=curve_names)
        #Set null value in log to Nan value using fancy index
        df1[df1 == null_value] = np.NaN
        
        def try_column_conversion(column):
            try:
                return column.astype(float)
            except ValueError:
                return column  # If conversion fails, return the original column as string
            
        df1 = df1.apply(try_column_conversion, axis=0)
        
        #Create a new column "Well_name" with the name of the well
        df1['Well_name'] = well_name
        df1['Well_name'] = df1['Well_name'].astype(str) ###############################################################################
        #Convert row number index to a column with reset_index, then set index 
        #as depth column with set_index. Do this operation 'in place', i.e.
        #do not create a new object. Keep a column 'DEPT' in the dataframe        
        ###THIS LINE SETS AN AUTOMATIC MULTIINDEX. USE ASCII DATA HELPER FUNCTIONS
        
        #Check to see if there are aliases for 'DEPT'. If so, create column
        #'DEPT' and delete the old column
        if 'DEPTH' in df1.columns:
            df1['DEPT'] = df1['DEPTH']
            del df1['DEPTH']
            df1 = df1.reset_index().set_index(['Well_name','DEPT'])# inplace=True) #14FEB2024 - Trying to manually set this multiindex anyways....
        elif 'MD' in df1.columns:
            df1['DEPT'] = df1['MD']
            del df1['MD']            
            df1 = df1.reset_index().set_index(['Well_name','DEPT'])# inplace=True) #14FEB2024 - Trying to manually set this multiindex anyways....
        #df1 = df1.reset_index().set_index('DEPT', drop=False)# inplace=True)
        #Assign name 'Row_number' for first column (Called index by default)
        df1.columns.values[0] = 'Row_number'
        
        setattr(self, 'A', df1)





    
    def assign_formation_tops(self):
        #Retrueve filepath for formation tops from Log object attribute
        tops_fn = self.tops_filename
        #Retrieve well name from Log object attribute
        well_name = self.W.UWI.value
        #Retrieve ASCII data from log object
        curve_data = self.A
        #Retrieve depth step value for ASCII data
        step = float(self.W.STEP.value)
        formation_tops = self.get_formation_tops(tops_fn, step)
        #Find the integer position of the column 'Formation'
        if formation_tops.empty:
            print ('{} does not have formation tops').format(well_name)
            self.set_multiindex()
            return None
        #Find the location of the formation top name column, called 'Form Alias'
        Fmn_col_loc = formation_tops.columns.get_loc('Form Alias')
        #Iterate over 'Depth' column in dataframe, find the same depth in the
        #curve_data data frame and assign the formation name of the top to the
        #corresponding depth in the curve_data dataframe
        for i,j in enumerate(formation_tops['Top MD']):
            curve_data.loc[j,'Fmn_top'] = formation_tops.iloc[i, Fmn_col_loc]
        #Forward fill the 'Fmn_top' column
        curve_data['Fmn_top'] = curve_data['Fmn_top'].ffill()
#        curve_data['Fmn_top'] = curve_data['Fmn_top'].ffill(inplace=True)
        self.set_multiindex()
        
    def set_multiindex(self):
        """Set the ASCII data for the log to a multi-index dataframe"""
        self.A.set_index(['Well_name', 'DEPT'], drop=False, inplace=True)

    def get_formation_tops(self, tops_fn, step):
        #Read CSV table with tops into a dataframe
        all_tops_df = pd.read_csv(tops_fn)
        #Retrieve the UWI from the well object to facilitate fancy indexing
        #of tops dataframe
        well_name = getattr(self.W, 'UWI').value
        #Use fancy index to work just with tops from the well with name
        #'well_name'
        tops_df = all_tops_df[all_tops_df['Well ID'] == well_name]
        #Use step value to properly round tops in the dataframe to the correct step
        tops_df['Top MD'] = tops_df['Top MD'].apply(lambda x: round(x / step) * step)
        return tops_df
        
    def slice_curve_interval(self, curve_data_df, depth_interval):
        """
        Take a curve and a tuple with two depths and slice out the data that
        corresponds to that depth interval
        """
        well_name = self.name
        top_depth = depth_interval[0]
        bottom_depth = depth_interval[1]
               
        sliced_df = curve_data_df.loc[(well_name, top_depth):(well_name, bottom_depth)]

        return sliced_df

    
    def create_curve_plot(self, depth_int=None, *curve_args):
        """Create a graph of curve data as a matplotliu plot"""
        #Assign curve data to local variable
        raw_curve_data = self.A
#        #Assign curve information data to local variable
#        curve_info = self.C
        #Calculate top and base of plots
        if depth_int  == None:
            curve_data = raw_curve_data
            #top, base = self.curve_limits()
        else:
            assert type(depth_int) == tuple
            top = min(depth_int)
            base= max(depth_int)
        
            depth_int_tuple = (top, base)
        
            curve_data = self.slice_curve_interval(raw_curve_data, depth_int_tuple)
        
        #The index of the curve_data dataframe cdefines the depths that will be 
        #used in the plots
        curve_depths = curve_data.index.get_level_values('DEPT').tolist()
        #The number of curves to plot is equal to the curves that are passed into
        #the function arguments
        
        num_curves = len(curve_args)
        
        #Assign figure and tuplle of axes to fig, axes variables
        fig, axes = plt.subplots(1, num_curves, sharey=True)
    
        print(f'Axes is {axes}')
        
        def generate_curves(figure, axes, index):
            curve_name = curve_args[index]
            curve = curve_data[curve_name] #.tolist()
            x_label_descr = getattr(self.C, curve_name).description
            x_label_unit = getattr(self.C, curve_name).unit
            
            x_label = ('{} ({})').format(x_label_descr, x_label_unit)
            
            y_label_descr = self.C.raw_data['DEPT'].description
            y_label_unit = self.C.raw_data['DEPT'].unit
                                              
            y_label = ('{} ({})').format(y_label_descr, y_label_unit)
            
            axes.plot(curve, curve_depths)
            axes.set_title(curve_name)
            axes.set_xlabel(x_label)
            axes.set_ylabel(y_label)
            
            #If the word 'ohm' is in the unit of this curve, set the x
            #scale of that  axes to logarithmic
            if 'ohm' in x_label.lower():   
                axes.set_xscale('log')
            
            
        #If there is only 1 curve being plotted, no loop
        if num_curves == 1:
            generate_curves(fig, axes, 0)                    
        #If more than 1 curve, Loop over the axes array and plot one curve on each axes
        else:            
            for i, ax in enumerate(axes):
                generate_curves(fig, ax, i)
                    
        #Invert y axis of Axes object so that depths are displayed shallowest
        #at the top and deepest at the bottom
        plt.gca().invert_yaxis()
        
        #TODO
        #Old return used to be just axes
        return fig, axes
                
    def get_curve_limits(self):
        """Find the depth limits of a particular curve"""
        curve_data = self.A
        curve_depths = curve_data.index.get_level_values(1) #Get only depths from multiindex
        top = min(curve_depths)
        base = max(curve_depths)
        
        return top, base
    
    def add_delta_PHIN_PHID(self):
        """Add a curve with the calculated difference between neutron and density porosity"""
        if ('PHIN' in self.C.mnems) and ('PHID' in self.C.mnems):
            self.A['DELTA_PHIN_PHID'] = self.A['PHIN'] - self.A['PHID']
        else:
            print ('Curves missing in {}').format(self.W.UWI.value)
    
    def test_nan(self, entry):
        """Test to see if there are "nan"s by converting objects to strings and then testing if string equals 'nan'"""
        if str(entry) != 'nan':
            return True
        else: 
           return False
            
    def assign_xy_coord(self, xy_fn):
        """Assign an XY coordinate to the well object if none exists, using an x-y file and checking to see what the coordinate pair is for that
        particular UWI. This will set an attribute in the "Well information" section of the log object
        """
    
        df1 = pd.read_csv(xy_fn, index_col='Well_name')
        UWI = self.W.UWI.value
        index = df1.index.tolist()
        if UWI in index and self.test_nan(df1.loc[UWI]['X_Coord']) and self.test_nan(df1.loc[UWI]['Y_Coord']):
            X_Coord = float(df1.loc[UWI]['X_Coord'])
            Y_Coord = float(df1.loc[UWI]['Y_Coord'])
        else:
            X_Coord = None
            Y_Coord = None
        setattr(self.W, 'XPOS', X_Coord)
        setattr(self.W, 'YPOS', Y_Coord)
        

    def forward_fill_curves(self, list_of_ffill_curves):
        """Forward fill curve data in the ASCII data dataframe. Useful in case
        depth steps between columns are unequal"""
        
        ascii_log_data = self.A
        
        for curve in list_of_ffill_curves:
            try:
                curve_data = ascii_log_data.loc[curve]
            except KeyError:
                continue
            
            curve_data.ffill(axis=1, inplace=True)
            
            
    def create_new_curve(self, function, new_column_name='New_Column'):
        """Use the dataframe.apply() method with axis=1 (applies function by row)
        to create a new column in the dataframe. The passed function should have
        the proper labels corresponding to the dataframe to correctly calculate
        the desired values

        Parameters
        ----------
        log_object : Object of the Log class
            Generated in the Plastic module.
        function : function
            The function to be applied row-wise to the dataframe. Lambda functions
            work. E.g. test_function = lambda row: ((row['C3_GAS'] * row['C2_GAS']) 
            / row['C1_GAS'])
                
        new_column_name : str, optional
            The name to give to the new column

        Returns
        -------
        The operations are performed in-place on the dataframe, no return is
        necessary
        """
        try:
            dataframe = self.A
            
            dataframe[new_column_name] = dataframe.apply(function, axis=1)
        except KeyError:
            well_name = self.name
            print(f'Not all columns were available in the {well_name} well')
            pass
    
    def drop_curve(self, curve_name):
        """Delete an unwanted column in the ASCII data

        Parameters
        ----------
        curve_name : str
            The name of the curve or column to drop.

        Returns
        -------
        None.

        """
        
        
        ascii_log_data = self.A
        
        #Perform the drop in place on the "curve_name" column
        ascii_log_data.drop(curve_name, axis='columns', inplace=True)
        
        
class Well_Section(Log):
    
    """A Log object is comprised of well sections. Each well section is comprised of attributes. All sections in a LAS file can be stored
    as a well section object, except for ASCII data, which is treated separately.
    """
    
    def __init__(self, section_data):
        #self.raw_data is an ordered dictionary of keys for section, 
        #then Well_Attribute objects
        self.raw_data = section_data
        self.mnems = self.get_mnems()
        self.descriptions = self.get_descriptions(section_data)
        
        self.assign_attributes()
    
    #List comprehension to take out all of mnem attribute values in the ordered
    #dict
    def get_mnems(self):
        return [self.raw_data[i].mnem for i in self.raw_data]
    
    #List comprehension to take out all of description attribute values in the
    #ordered dict
    def get_descriptions(self, section_data):
        return [self.raw_data[i].description for i in self.raw_data]
    
    def assign_attributes(self):
        for i in self.raw_data:
            setattr(self, i, self.raw_data[i])
            
        def __repr__(self):
           print(self.raw_data)
    
class Well_Attribute(Well_Section):
    
    #TODO
    #Make Well_attribute a sub-class of well section?
    
    """A well attribute is a "building block" of a Well_Section object. This class definition specifies the main
    constituents of a well attribute that are specified in the CWLS LAS standard
    """
    
    def __init__(self, *data):
        mnem, unit, value, description = data
        self.mnem = mnem
        self.unit = unit
        self.value = value
        self.description = description
        
        #print('Attribute values: ', self.mnem, self.unit, self.value, self.description)
    
    #Return the value attribute of the Well_Attribute object
    def __getitem__(self):
        return self.value
    
#    def __getattribute__(self):
#        return self.value
    
    #Print the 4 attributes of the Well_Attribute object when the attribute is accessed directly
    def __repr__(self):
       return ('{} {} {} {}').format(self.mnem, self.unit, self.value, self.description)
   
def well_slice_perf_interval_data(well, perf_intervals):
    well_df = well.A.loc[float(perf_intervals[0][0]):float(perf_intervals[0][1])]
    if len(perf_intervals) > 1:
        for i in range(len(perf_intervals))[1:]:
            temp_df = well.A.loc[float(perf_intervals[i][0]):float(perf_intervals[i][1])]
            well_df = pd.concat([well_df, temp_df], axis=0)
    
    return well_df
 
def forward_fill_curves(df, list_of_ffill_curves='All'):
    """Forward fill curve data in the ASCII data dataframe. Useful in case
    depth steps between columns are unequal"""
    
    #Get list of index names
    index_list = list(df.index.names)
    
    #Reset index will take index and make them into columns in the dataframe
    df = df.reset_index()
    
    # print(df)
     
    #If nothing is specified in the list of columns, forward fill all columns
    if list_of_ffill_curves == 'All':

        forward_fill_df = df.ffill(limit=3)
        
    #Set the index back to the original index
    
    # print(forward_fill_df)
    
    forward_fill_df = forward_fill_df.set_index(index_list)
         
    return forward_fill_df


def find_closest_depth_value(df, value):
    """Get the integer index of the depth value in the dataframe that is
       closest to the value passed to the function
    

    Parameters
    ----------
    df : DataFrame
        DataFrame of Log ASCII data.
    value : int or float
        The value that is being searched for.

    Returns
    -------
    min_delta_label : int or float
        The depth in the index that is closest to the value being searched \
            for.
    min_delta_loc : int
        The integer position of the row corresponding to the depth that is \
            closest to the value being searched for.

 """
    try:
        assert type(value) == int or float        
    except AssertionError:
        incorrect_type = str(type(value))
        print(f'An incorrect type, {incorrect_type} was passed. Type must be a positive int or float')
        return
    
    try:
        assert value > 0
    except AssertionError:
        print(f'A negative number, {value} was passed. Numbers must be positive')
        return
        
    
    #Create a series with the depth data from the multi-index
    df_reset_index = df.reset_index()
    depth_series = df_reset_index['DEPT']
    
    #Suptract the specified value from the values in the series and calculate
    #absolute value to find smallest residual
    depth_series_delta = (depth_series - value).abs()
    
    #Returns the label from the index that corresponds to the minimum value
    min_delta_label = depth_series_delta.idxmin()
    
    #Returns the integer position of the label    
    min_delta_loc = depth_series_delta.index.get_loc(min_delta_label)
    
    #The depth corresponding to the integer position of the label that was
    #returned
    min_delta_depth = depth_series.iloc[min_delta_loc]
    
    #Check if the values returned were the first or the last values, which
    #likely indicates an undesired result
    depth_series_first_value = depth_series.iloc[0]
    depth_series_last_value = depth_series.iloc[-1]
    
    if min_delta_depth == depth_series_first_value:        
        print(f'WARNING. The closest depth was the shallowest value,\
              {depth_series_first_value}. Please ensure this is correct')
              
    if min_delta_depth == depth_series_last_value:        
        print(f'WARNING. The closest depth was the deepest value, \
              {depth_series_last_value}. Please ensure this is correct')
              
    #Return label and integer position of matched value
    return (min_delta_depth, min_delta_loc)
             
def round_depth_values(df, depth_step, depth_index='DEPT'):
     """Resample a dataframe (from a log object, for example) with a particular
     depth step. 

     Parameters
     ----------
     df : DataFrame
         DataFrame with ASCII log data (eg. the ".A" component of a Log object)
     depth_step : float
         The depth step to which data should be resampled
     depth_index : str
         The name of the column corresponding to the depth value

     Returns
     -------
     Ascii log data DataFrame
     
     """
     #Get list of index names
     index_list = list(df.index.names)
     
     #Reset index will take index and make them into columns in the dataframe
     df = df.reset_index()
     
     #Use step value to properly round tops in the dataframe to the correct step
     df[depth_index] = df[depth_index].apply(lambda x: round(x / depth_step) * depth_step)
     
     df = df.set_index(index_list)
     
     return df
      
def resample_depth_step_pruning(df, depth_step=0.5, depth_index='DEPT'):
    """Apply depth step resampling and 'prune' duplicate values"""
    
    #Get list of index names
    index_list = list(df.index.names)
    
    #Round values to indicated depth step
    rounded_depth_df = round_depth_values(df, depth_step)
    
    #Reset index to get index names into columns
    rounded_depth_df = rounded_depth_df.reset_index()
    
    #Drop the duplicates in the depth index column and keep the first entry
    rounded_depth_df = rounded_depth_df.drop_duplicates(depth_index)
    
    #Set the index back to the original index
    rounded_depth_df = rounded_depth_df.set_index(index_list)
    
    #Drop the 'Row_number' column which is an artefact of reset_index and
    #set_index
    #rounded_depth_df.drop('Row_number', axis=1, inplace=True)
     
    return rounded_depth_df
             
def forward_fill_and_resample_pruning(df, depth_step=0.5, depth_index='DEPT'):
    """
    Apply forward fill of curves in an log ascii data dataframe and resample
    at a specified depth step.

    Parameters
    ----------
    df : DataFrame
        Log ascii data dataframe
    depth_step : float, optional
        The depth step at which resampling should occur. The default is 0.5.
    depth_index : str, optional
        The index name for depth in the dataframe. The default is 'DEPT'.

    Returns
    -------
    df : DataFrame
        The forward-filled and resampled DataFrame.

    """
    
    df = forward_fill_curves(df)
    
    df = resample_depth_step_pruning(df)
    
    return df




def create_new_curve_log(log_object, function, new_column_name='New_Column'):
    """Use the dataframe.apply() method with axis=1 (applies function by row)
    to create a new column in the dataframe. The passed function should have
    the proper labels corresponding to the dataframe to correctly calculate
    the desired values

    Parameters
    ----------
    log_object : Object of the Log class
        Generated in the Plastic module.
    function : function
        The function to be applied row-wise to the dataframe. Lambda functions
        work. E.g. test_function = lambda row: ((row['C3_GAS'] * row['C2_GAS']) 
        / row['C1_GAS'])
            
    new_column_name : str, optional
        The name to give to the new column

    Returns
    -------
    The operations are performed in-place on the dataframe, no return is
    necessary
    """
    #Error handling
    try:
        df = log_object.A
        
        #Use the apply method to run function across all column rows
        df[new_column_name] = df.apply(function, axis=1)
    except KeyError:
        well_name = log_object.name
        print(f'Not all columns were available in the {well_name} well')
        pass


def create_logarithmic_gas_res_curves_df(df, replace_neg_inf=True):
    """
    Add the logarithmic "_LOG10" gas and resistivity curves to a given dataframe with gas curve
    data

    Parameters
    ----------
    df : DataFrame
        Source ASCII log data.

    Returns
    -------
    df : DataFrame
        DataFrame with additional "_LOG10" curves added.

    """
    curves_to_convert = ['RESD','RESM','RESS','C2_GAS','C3_GAS','C1_GAS', 
                         'IC4_GAS', 'IC5_GAS', 'NC4_GAS', 'NC5_GAS', 'TOTALGAS']

    for curve in curves_to_convert:
        log10_curve_name = f'{curve}_LOG10'
        df[log10_curve_name] = df[curve].apply(np.log10)
    
    if replace_neg_inf == True:
        #Get object for negative infinity
        neg_inf = np.log10(0)
        
        #Replace negative infinity with np.nan
        df = df.replace(neg_inf, np.nan)
    
    return df

def create_PHIDSS_RHOB_CALC(df):
    df['PHIDSS_RHOB_CALC'] = df.apply(lambda x: (2.65 -x['RHOB'])/1.65, axis=1)
    
    return df


def create_delta_PHIN_PHIDSS(df):
    df['DELTA_PHID_PHIN'] = df.apply(lambda row: row['PHIDSS_RHOB_CALC'] - row['NPHI'], axis=1)
    
    return df
    

def create_logarithmic_curve(log_object, target_curve_name, mode='base10'):
   
    """This function takes a column of a dataframe and creates a new column
    in the dataframe with the transform applied
    
    Parameters
    ----------
    log_object : Log class object
        The target log object.
    target_curve_name : str
        The name of the curve upon which the operation will be performed.
    mode : str, optional
        Specifies whether the logarithm is taken in base 10, or as a natural 
        logarithm. The default is base10. Use 'natural' to activate natural
        logarithm.

    Returns
    -------
    None.

    """
    
    #Error handling and assignment of target curve
    try:
        df = log_object.A
        target_curve = df[target_curve_name]
       
    except KeyError:
        well_name = log_object.name
        print(f'Not all columns were available in the {well_name} well')
        pass
    
    #Specifty the suffix and function used based upon the 'mode' parameter
    mode_dict = {'base10': ('_LOG10', (np.log10)), 'natural':('LN', np.log)}
    
    #Unpack the siffix and mode from the dictionary entry
    mode_suffix, mode_function = mode_dict[mode]
    
    #Specify new column name
    new_column_name = target_curve_name + mode_suffix
    
    #Create new column
    df[new_column_name] = target_curve.apply(mode_function)
    
def create_window_curve(log_object, target_curve_name, depth_steps, mode='mean'):
    #Error handling and assignment of target curve
    try:
        df = log_object.A
        target_curve = df[target_curve_name]
       
    except KeyError:
        well_name = log_object.name
        print(f'Not all columns were available in the {well_name} well')
        pass
    
    window_size = (depth_steps * 2) + 1
    
    mode_dict = {'sum': ('_sum', pd.Series.sum),
                 'mean': ('_mean', pd.Series.mean),
                 'median': ('_median', pd.Series.median),
                 'min': ('_min', pd.Series.min),
                 'max': ('_max', pd.Series.max),
                 'std': ('_std', pd.Series.std),
                 'var': ('_var', pd.Series.var)
                 }
    
        
    new_column_name = target_curve_name + '_' + mode + '_' + str(depth_steps)
    
    df[new_column_name] = target_curve.rolling(window=window_size, center=True, min_periods=1).apply(mode_function)
    
    
def wetness_calc(row):
    """
    Calculate the Haworth wetness gas ratio based on a row passed in from a
    dataframe. This function is defined to work with the DataFrame.apply()
    method

    Parameters
     ----------
    row : pd.Series
        The pandas Series object corresponding to a row in a dataframe.

    Returns
    -------
    float
        The value corresponding to the wetness ratio. If there is a divide
        by zero error, the function will return np.nan

    """
    
    #Ensure that the denominator in the equation is not equal to zero
    if ((row['C1_GAS'] != 0) & (row['C2_GAS'] != 0) & (row['C3_GAS'] != 0) &
        (row['IC4_GAS'] != 0) & (row['NC4_GAS'] != 0) &( row['IC5_GAS'] != 0) &
        (row['NC5_GAS'] != 0)):
     
     
        wetness = ((row['C2_GAS'] + row['C3_GAS'] + row['IC4_GAS'] + 
                    row['NC4_GAS'] + row['IC5_GAS'] + row['NC5_GAS']) / 
                   (row['C1_GAS'] + row['C2_GAS'] + row['C3_GAS'] + 
                    row['IC4_GAS'] + row['NC4_GAS'] + row['IC5_GAS'] +
                    row['NC5_GAS'])) * 100
         
        return wetness
    
    else:
        return np.nan

def balance_calc(row):
    """
    Calculate the Haworth balance gas ratio based on a row passed in from a
    dataframe. This function is defined to work with the DataFrame.apply()
    method

    Parameters
     ----------
    row : pd.Series
        The pandas Series object corresponding to a row in a dataframe.

    Returns
    -------
    float
        The value corresponding to the balance ratio. If there is a divide
        by zero error, the function will return np.nan

    """
    
    #Ensure that the denominator in the equation is not equal to zero
    if ((row['C3_GAS'] != 0) & (row['IC4_GAS'] != 0) & (row['NC4_GAS'] != 0) &
        (row['IC5_GAS'] != 0) & (row['NC5_GAS'] != 0)):
        
        balance = ((row['C1_GAS'] + row['C2_GAS']) / (row['C3_GAS'] + row['IC4_GAS']
                    + row['NC4_GAS'] + row['IC5_GAS'] + row['NC5_GAS']))
             
        return balance
    
    else:
        return np.nan

def character_calc(row):
    """
    Calculate the Haworth character gas ratio based on a row passed in from a
    dataframe. This function is defined to work with the DataFrame.apply()
    method

    Parameters
    ----------
    row : pd.Series
        The pandas Series object corresponding to a row in a dataframe.

    Returns
    -------
    float
        The value corresponding to the character ratio. If there is a divide
        by zero error, the function will return np.nan

    """
    
    #Ensure that the denominator in the equation is not equal to zero
    if row['C3_GAS'] != 0:
        character = ((row['IC4_GAS'] + row['NC4_GAS'] + 
                      row['IC5_GAS'] + row['NC5_GAS'])
                     / row['C3_GAS'])
        
        return character
    else:
        #pd.NA is the pandas generic value for missing data
        return np.nan

def apply_fluid_classifier_haworth(df):
    """
    Pass a dataframe in and apply the Haworth fluid classification
    

    Parameters
    ----------
    df : DataFrame
        Curve data dataframe.

    Returns
    -------
    df: DataFrame
        Curve data dataframe

    """
    if 'Haworth_fluid_classification' not in df.columns:
        df['Haworth_fluid_classification'] = None
        
    df = df.apply(fluid_classifier_haworth, axis=1)
    
    return df

def fluid_classifier_haworth(row):

    
    wetness = row['WETNESS_RATIO']
    balance = row['BALANCE_RATIO']
      
    
    if balance > 100:
        if wetness <0.5: 
            row['Haworth_fluid_classification'] = 'Very Dry Gas - Non Productive'
        else:
            row['Haworth_fluid_classification'] = 'Very Dry Gas - Possibly Productive'
    elif 0.5 < wetness < 17.5:
        if balance > wetness:
            row['Haworth_fluid_classification'] = 'Gas'
    elif 17.5 < wetness < 40:
        if balance < wetness:
            row['Haworth_fluid_classification'] = 'Oil'
    elif wetness > 40:
        row['Haworth_fluid_classification'] = 'Residual Oil'
            
    return row        

    
def calculate_estimated_temperature(row):
    
    feet_to_km = 3208 #ft/km
    
    depth_ft = row.index.get_level_values('DEPT')
    depth_km = depth_ft / feet_to_km
    geothermal_gradient = 25 #deg C / km
    
    estimated_temperature = 25 + (depth_km * geothermal_gradient)
    
    return estimated_temperature

def calculate_PHIDSS(bulk_density):
    silica_density = 2.65 #g/cc
    water_density = 1 #g/cc
    
    PHIDSS = (silica_density - bulk_density) / (silica_density - water_density)
    
    return PHIDSS
                                        
def calculate_apparent_water_resistivity(row):
    #Archie equation for apparent water resistivity
    
    total_resistivity = row['RESD']
    
    bulk_density = row['RHOB']
    
    PHIDSS = calculate_PHIDSS(bulk_density)
            
    tortuosity_factor = 1 #Also known as 'a'
    cementation_exponent = 2 #Also known as 'm'
    
    Rwa = (total_resistivity*(PHIDSS**cementation_exponent))/tortuosity_factor
    
    return Rwa
    

def calculate_rate_of_change_curve(well_data_df, curve_name, num_depth_steps=13):
    
    
    #TODO figure out how to leverage the well attributes to automatically
    #return the depth step for a Log object. The STEP attribute may need to be
    #updated after merging the curves for this to work properly.
    
    #Crudely calculate the depth step by subtracting the second value in the
    #depth index from the third value in the depth index. 
    depth_values = well_data_df.index.get_level_values('DEPT')
    depth_step = depth_values[2] - depth_values[1]
    
    curve = well_data_df[curve_name]
    shifted_curve = curve.shift(num_depth_steps)
    
    difference_curve = curve - shifted_curve
    
    rate_curve = difference_curve / (num_depth_steps * depth_step)
    
    rate_curve_name = f'{curve_name}_rate'
    
    well_data_df[rate_curve_name] = rate_curve
    
def plot_derivative_log_data(df, steps=3, screening_formation=None):
    if screening_formation != None:
        df = df[df['Strat_unit_name'] == screening_formation]
    
    
    rhob_series = df['RHOB_rate']
    gr_series = df['GR_rate']
    resdlog10_series =  df['RESD_LOG10']
    oil_show_series = df['OIL_SHOW']
    C3_gas_show_series = df['C3_GAS_LOG10']
    
    plt.figure(figsize=(40,20))
    plt.scatter(rhob_series, resdlog10_series, c=C3_gas_show_series, cmap='viridis', s=0.3)
    plt.xlim(-0.5,0.5)
    plt.ylim(-1.2,1.2)
    figure_name = f'RHOB_ResDLog10_{steps}_steps_{screening_formation}_fmn.png'
    
    plt.savefig(figure_name)
    
    plt.figure(figsize=(40,20))
    plt.scatter(gr_series, resdlog10_series, c=C3_gas_show_series, cmap='viridis', s=0.3)
    plt.xlim(-100,100)
    plt.ylim(-1.2,1.2)
    figure_name = f'GR_ResDLog10_{steps}_steps_{screening_formation}_fmn.png'
    
    plt.savefig(figure_name)
    
    
def scatter_hist(x, y, ax, ax_histx, ax_histy, color_data=None):
    # no labels
    ax_histx.tick_params(axis="x")#, labelbottom=True)
    ax_histy.tick_params(axis="y")#, labelleft=True)
    
    if len(color_data > 0):
        ax.scatter(x, y, s=0.5, c=color_data, cmap='viridis')
    else:
        ax.scatter(x, y, s=0.5)
    # the scatter plot:
    
    
    num_bins = 500
    ax_histx.hist(x, bins=num_bins, histtype='bar', stacked=True)
    ax_histy.hist(y, bins=num_bins, orientation='horizontal', histtype='bar', stacked=True)
    
def plot_derivative_log_data_with_histogram(df, steps=3, screening_formation=None):
    if screening_formation != None:
        df = df[df['Strat_unit_name'] == screening_formation]
    
    
    rhob_series = df['RHOB_rate']
    gr_series = df['GR_rate']
    resdlog10_series =  df['RESD_LOG10_rate']
    oil_show_series = df['OIL_SHOW']
    C3_gas_show_series = df['C3_GAS_LOG10']
    
    figure_size=(40,20)
    
    fig = plt.figure(layout='constrained', figsize=figure_size)
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    scatter_hist(rhob_series, resdlog10_series, ax, ax_histx, ax_histy, oil_show_series)
    plt.xlim(-0.5,0.5)
    plt.ylim(-1.2,1.2)
    figure_name = f'RHOB_ResDLog10rate_{steps}_steps_{screening_formation}_fmn_histogram.png'
    plt.savefig(figure_name)
    
    
    fig = plt.figure(layout='constrained', figsize=figure_size)
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    scatter_hist(gr_series, resdlog10_series, ax, ax_histx, ax_histy, oil_show_series)
    plt.xlim(-100,100)
    plt.ylim(-1.2,1.2)
    figure_name = f'GR_ResDLog10rate_{steps}_steps_{screening_formation}_fmn_histogram.png'
    plt.savefig(figure_name)   
    
   
    
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
    
    
def create_new_curve(log_object, function, new_column_name='New_Column'):
    """Use the dataframe.apply() method with axis=1 (applies function by row)
    to create a new column in the dataframe. The passed function should have
    the proper labels corresponding to the dataframe to correctly calculate
    the desired values

    Parameters
    ----------
    log_object : Object of the Log class
        Generated in the Plastic module.
    function : function
        The function to be applied row-wise to the dataframe. Lambda functions
        work. E.g. test_function = lambda row: ((row['C3_GAS'] * row['C2_GAS']) 
        / row['C1_GAS'])
            
    new_column_name : str, optional
        The name to give to the new column

    Returns
    -------
    The operations are performed in-place on the dataframe, no return is
    necessary
    """
    try:
        dataframe = log_object.A
        
        dataframe[new_column_name] = dataframe.apply(function, axis=1)
    except KeyError:
        well_name = log_object.name
        print(f'Not all columns were available in the {well_name} well')
        pass
