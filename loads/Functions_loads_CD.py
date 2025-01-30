import numpy as np
from numpy import cos,sin
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.fftpack import *
import scipy as sp
import scipy.signal
import scipy.signal as signal
from scipy.optimize import curve_fit
import sys
import time
import glob
import netCDF4 as nc
import xarray as xr
import pickle
import pvlib
import os
from nptdms import TdmsFile
from datetime import datetime
import pytz


#%% Read data


# Read loads raw data

# def read_tdms_file_into_dataframe(tdms_filename):
#     try:
#         tdms_file = TdmsFile(tdms_filename)
#         data = {}
#         for group in tdms_file.groups():
#             for channel in group.channels():
#                 data[channel.name] = channel[:]
#         df = pd.DataFrame(data)
#         return df
#     except Exception as e:
#         print(f"Error: {e}")
        
        
def read_tdms_file_into_dataframe(tdms_filename, verbose = False):
    
    #tdms_filename = 'Y:\Wind-data/Restricted/Projects/NSO/CrescentDunes-Loads/SlowData/'+'CrescentDunes_2024_03_13_08_28_55_1Hz.tdms'
    
    try:
        tdms_file = TdmsFile(tdms_filename)
        data = {}
        for group in tdms_file.groups():  # only one group
            
            if verbose:
                with open("loads_channels.txt", "w") as f:
                    f.write(str(group) + "\n")    
            
            for channel in group.channels():  # channels correspond to variables
                data[channel.name] = channel[:]
                
                if verbose:
                    print (channel)
                    with open("loads_channels.txt", "a") as f:
                        f.write(str(channel) + "\n")                    
                    for property_name, property_value in channel.properties.items():
                        print(f"  {property_name}: {property_value}")
                        with open("loads_channels.txt", "a") as f:
                            f.write(f"  {property_name}: {property_value}"  + "\n")
                
        df = pd.DataFrame(data)
        
        
        ### Fix time stamp
        
        # Ensure 'MS Excel Timestamp' is present in each DataFrame
        if 'MS Excel Timestamp' not in df.columns:
            print(f"Warning: DataFrame from {tdms_file} does not contain 'MS Excel Timestamp' column.")
            pass
        if 'LabVIEW Timestamp' not in df.columns:
            print(f"Warning: DataFrame from {tdms_file} does not contain 'LabVIEW Timestamp' column.")
            pass
        
        df.index = pd.to_datetime(df['LabVIEW Timestamp'], unit="s", origin=pd.Timestamp("1904-01-01"))

        return df
    except Exception as e:
        print(f"Error: {e}")
        
def read_multiple_tdms_file_into_dataframe(file_list, resample_freq = False):


    dfs = []

    for tdms_file in file_list:
        
        print(tdms_file)
        

        df = read_tdms_file_into_dataframe(tdms_file)
        
        if resample_freq != False:
            df = df.resample(resample_freq).mean()

        dfs.append(df)
    
    # Concatenate DataFrames along 'MS Excel Timestamp' axis
    concatenated_df = pd.concat(dfs, axis=0, ignore_index=False)
    
    concatenated_df = concatenated_df.sort_index()
    
    return concatenated_df        
        


def process_in_same_directory():

    zip_file_path =  'Y:\Wind-data/Restricted/Projects/NSOCrescentDunes-Loads/FastData/'
    
    dfs = []
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        for file_name in file_list:
            if file_name.endswith('.tdms'):
                with zip_ref.open(file_name) as tdms_file:
                    df = read_tdms_file_into_dataframe(tdms_file)
                    # Ensure 'MS Excel Timestamp' is present in each DataFrame
                    if 'MS Excel Timestamp' not in df.columns:
                        print(f"Warning: DataFrame from {file_name} does not contain 'MS Excel Timestamp' column.")
                        continue
                    print (file_name)
                    print (df['MS Excel Timestamp'][0])
                    dfs.append(df)
    
    # Concatenate DataFrames along 'MS Excel Timestamp' axis
    concatenated_df = pd.concat(dfs, axis=0, ignore_index=True)
    return concatenated_df





def read_and_combine_pickle_files(inflow_files):
    dataframes = []

    for file_path in inflow_files:
        try:
            inflow = pd.read_pickle(file_path)
            inflow = inflow.sort_index()
            dataframes.append(inflow)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=False)
        combined_df = combined_df.sort_index()  # Ensure the final combined DataFrame is sorted by index
        return combined_df
    else:
        print("No valid dataframes to combine.")
        return None
        
    


# SCADA data


# Define a custom date parser depending on daylight time or standard time (Pacific)
def parse_datetime_with_timezone(date_str):
    # Remove leading/trailing spaces
    date_str = date_str.strip()
    
    # Check for PDT or PST in the string
    if date_str.endswith('PDT'):
        dt = datetime.strptime(date_str, '%d-%b-%Y %H:%M:%S.%f PDT')
        # Assign Pacific Daylight Time
        timezone = pytz.timezone('America/Los_Angeles')
        return timezone.localize(dt, is_dst=True)  # is_dst=True ensures PDT
    elif date_str.endswith('PST'):
        dt = datetime.strptime(date_str, '%d-%b-%Y %H:%M:%S.%f PST')
        # Assign Pacific Standard Time
        timezone = pytz.timezone('America/Los_Angeles')
        return timezone.localize(dt, is_dst=False)  # is_dst=False ensures PST
    else:
        raise ValueError(f"Unrecognized timezone in timestamp: {date_str}")


  
def read_SCADA_file(paths, filename_search_strings, resample_freq = "1min"):
    
    
    # filename = "Report_14_MAR_2024*.csv"
    # path = '../SCADA/EMS1/'
    #vfile = glob.glob(os.path.join(path, filename))[0]
    
    # Initialize an empty list to store the file paths
    all_files = []
    
    # Loop through each path and filename search string
    for path in paths:
        for filename_search_string in filename_search_strings:
            files = glob.glob(os.path.join(path, filename_search_string))
            all_files.extend(files)  # Add files to the list
        
    files = all_files # glob.glob(os.path.join(path, filename_search_string))
    
    
    df1_list = []
    df2_list = []
    df3_list = []
    
   

    # Loop through the files and read each one into a dataframe
    for file in files:
        
        print(file)
        
        df = pd.read_csv(file, skiprows=[2], header=[0, 1], index_col=0, 
                         #parse_dates=True, 
                         #date_format='%d-%b-%Y %H:%M:%S.%f PDT ', 
                         #date_parser=custom_date_parser,
                         low_memory=False)
        
        # PArse timestamp
        df.index = df.index.to_series().apply(parse_datetime_with_timezone)
        
        # Convert the index from PDT to UTC
        #df.index = df.index.tz_localize('America/Los_Angeles').tz_convert('UTC')
        df.index = df.index.tz_convert('UTC')
        
        # Clean up the column names by stripping leading and trailing spaces
        df.columns = df.columns.map(lambda x: (x[0].strip(), x[1].strip()))
        
        # Drop columns that have "Hex" in the second row
        df = df.loc[:, df.columns.get_level_values(1) != '(Hex)']
        
        # Flatten the multi-level columns - or drop second header column
        #df.columns = df.columns.map(' '.join).str.strip()
        df.columns = df.columns.get_level_values(0)
        
        # Define numeric and string columns
        string_columns = ["State"]  #["Heliostat", 
        numeric_columns = [col for col in df.columns if col not in ["Heliostat", "State"]]
        
        # Define shared aggregation rules
        from collections import defaultdict 
        
        agg_rules = defaultdict(list)
        for col in numeric_columns:
            agg_rules[col] = ["mean"]  # Shared rules for numeric columns
        
        for col in string_columns:
            agg_rules[col] = ["first"]
        
        # Convert defaultdict back to a normal dictionary for agg()
        agg_rules = dict(agg_rules)
        
        # One datafile for each heliostat
        df1 = df[df['Heliostat'] == 'W2-74-11'].drop(columns=["Heliostat"])
        df2 = df[df['Heliostat'] == 'W2-73-25'].drop(columns=["Heliostat"])
        df3 = df[df['Heliostat'] == 'W2-58-16'].drop(columns=["Heliostat"])
        
        df1 = df1.resample(resample_freq).agg(agg_rules)
        df2 = df2.resample(resample_freq).agg(agg_rules)
        df3 = df3.resample(resample_freq).agg(agg_rules)        
        
        # for H in [df1, df2, df3]:
              
            # H = H.apply(pd.to_numeric, errors='coerce')
            # H = H.resample("1min").mean()
        
        df1_list.append(df1)
        df2_list.append(df2)
        df3_list.append(df3)
    
    # Concatenate all dataframes into one
    H1 = pd.concat(df1_list)
    H2 = pd.concat(df2_list)    
    H3 = pd.concat(df3_list)
    
    H1 = H1.sort_index()
    H2 = H2.sort_index()    
    H3 = H3.sort_index()    

    return H1, H2, H3




#%%% Processing

def loads_initial_postprocess(loads):
    
        
    # Add wind speed column
    loads["H1_Wind_Speed"] = (loads['H1_Wind Speed U']**2 + loads['H1_Wind Speed V']**2)**0.5
    loads.H1_Wind_Speed = loads.H1_Wind_Speed.where(loads.H1_Wind_Speed<100)
    
    
    # Clean up the column names by stripping leading and trailing spaces
    loads.columns = loads.columns.str.strip()
    
    # drop columns
    loads = loads.drop(columns=['MS Excel Timestamp','LabVIEW Timestamp','Scan Errors', 'Late Scans'])
    
    
    # resample (loads slow 1s data is samples at full seconds)
    # loads = loads.resample("S").first()
    
    loads = loads.dropna(how="all")
    
    return loads


def add_load_coefficients_CD(data, wind_speed_limit=3):
    
    """ 
    'data' must be a dataframe with combined loads and wind data
    'wind_speed_limit' is the limit above coefficients are calculated
    """
    
    L = 10.42          # m, length of Heliostat (shord width)
    W = 11.41          # m, width of Heliostat
    A = L*W
    Hc = 5.6   # m, height of pivot axis
    
    
    if 'rho' not in data.columns:
        
        try:
            data['rho'] = rho(data.p, data.RH, data.Temp).interpolate().rolling('60s', center=True, min_periods=0).mean()
        except Exception:
            data['rho'] = 1.29

    try:
        wspd = data.wspd_Mid.rolling('600s', center=True, min_periods=0).mean()  
            
        wspd = wspd.where(wspd> wind_speed_limit)  
        
        # dynamic pressure
        q = data.rho/2 * wspd**2
        
        # Lift coefficients
        for column in data[[col for col in data.columns if ('H1_F_Lift' in col)  & ('_m' not in col) & ('_s' not in col) & ('C' not in col)]]:
            data[column[:3]+"C_lift"] = data[column] *1000/ (q * A)   
        
        # # Bending moment coefficients
        # for column in data[[col for col in data.columns if ('Bending' in col)  & ('_m' not in col) & ('_s' not in col) & ('C' not in col)]]:
        #     data[column[:6]+"C_"+ column[6:]] = data[column] *1000/ (q * L  * W * Hc)   
        # # Torque moment coefficients
        # for column in data[[col for col in data.columns if ('Torque' in col)  & ('_m' not in col) & ('_s' not in col) & ('C' not in col)]]:
        #     data[column[:6]+"C_"+ column[6:]] = data[column] *1000/ (q * L * W**2)   
        # # Drag force coefficients
        # for column in data[[col for col in data.columns if ('Bending' in col)  & ('_m' not in col) & ('_s' not in col) & ('C' not in col)]]:
        #     fx = data[column] *1000 / Hc
        #     data[column[:6]+"Cfx"] = fx / (q * L  * W)  
            
        # Exclude unreasonably high values
        for column in data[[col for col in data.columns if ( ('C_' in col) or ('Cfx' in col))]]:
            data[column] = data[column].where(abs(data[column])<30)
        
    except Exception:
        pass

    return data





#%%
if __name__ == "__main__":
    
    pass




