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
from numba import jit
import pvlib
import os


#%% Read data


# Gill w bug correction
def apply_factors_for_w_bug(value):
    if value >= 0:
        return value * 1.166 
    else:
        return value * 1.289



### Met tower

def add_microseconds(data,time_step=0.05):
    
    ### !!!! repai that timestamp always starts at zero seconds. !!!!
    
    ''' adds milliseconds with frequency of time_step 
    to the recorded data which have only second-resolved timestamp 
    '''
    time_diff = data['RECNBR'].diff() * time_step
    second = data.index.second # extract seconds from timestamp
    ms = np.zeros(len(second)) # initiate column with microseconds filled with zeros
    for i in range (1,len(second)): # create column with microseconds between 0 and 1000
        if second[i]==second[i-1]: 
            ms[i] = ms[i-1] + time_diff.iloc[i]*1000 
    data["orig_timestamp"] = data.index
    data.index = ( data.index + pd.to_timedelta(ms, unit='ms') ) # add microseconds to timestamp

    return data


def read_sonic(datafile):
    
    ''' 
    - reads data from inflow mast Sonic data file for Crescent Dunes data
    - makes data corrections (outliers, faulty values)
    '''
    
    # print (datafile)
    
    ## read data
    # if pd.to_datetime(datafile[-19:-9], format = "%Y_%m_%d") < pd.to_datetime('2024-03-17 00:00:00'):
    #     skip = [0]
    # else:
    #     skip = [0,2,3]
        
        
    skip = [0,2,3]
    data = pd.read_csv(datafile,
                    index_col = 0,
                    header = 0,    
                    skiprows = skip, 
                    engine = 'c',
                    on_bad_lines='warn', 
                    na_values = {'NAN', '.'},
                    dtype = float,
                    parse_dates=True
                        )
    
    if "RECORD" in data.columns:
        data = data.rename(columns={"RECORD": "RECNBR", "TIMESTAMP": "TMSTAMP"})
        
    # # Add milliseconds to timestamp    
    # if pd.to_datetime(datafile[-19:-9], format = "%Y_%m_%d") < pd.to_datetime('2024-03-17 00:00:00'):
    #     data = add_microseconds(data,time_step=0.05)
    # else:
    #     data["orig_timestamp"] = data.index
    #     data.index = pd.to_datetime(data.index, format='mixed')
    
    
    
    data["orig_timestamp"] = data.index
    data.index = pd.to_datetime(data.index, format='mixed')

    
    if 'RECNBR' in data:
        data.drop(['RECNBR'], axis=1, inplace=True)
        
    
    
    # plot for diagnostics of spikes
    # flag = 0
    # test = plt.figure()
    # for height_col in [col for col in data.columns if '_ax_' in col]:    
    #     if (data[height_col].max()>30) | (data[height_col].min()<-30):
    #         flag=1
    #         plt.plot(data[height_col], label=height_col+' '+datafile[-33:-27])
    # if flag==1:
    #     plt.legend(loc=1)
    #     plt.title('{}_to_{}.png'.format(data.index[0].date(), data.index[-1].date()))
    #     test.savefig('C:/Users/uegerer/Desktop/NSO/Faulty_mast_values/{}_{}h_to_{}_{}h_'.format(data.index[0].date(),data.index[0].hour, data.index[-1].date(), data.index[-1].hour )+datafile[-33:-27]+'.png')
    # plt.close(test)
        
    # filter spikes above 40 m/s (unrealistic, but present in data as spikes)
    for height_col in [col for col in data.columns if '_ax_' in col]:    
        data[height_col] = data[height_col].where(abs(data[height_col])<30)
        
    return data


def calc_wind(data):
    
    ''' 
    - adds wind direction and horizontal wind
    '''

    ## loop over every mast height for calculating wind direction and speed
    
    for height_col in [col for col in data.columns if 'U_ax_' in col]:    # loop over every mast height
        
        U = height_col
        V = height_col.replace("U_ax", "V_ax")
        col_dir = height_col.replace("U_ax", "wdir")
        col_spd = height_col.replace("U_ax", "wspd")

        # calculate wind direction and horizontal wind       
        data[col_dir] = np.degrees(np.arctan2( - data[V], data[U])) # this is the direction of the wind speed vector
        data[col_dir] = data[col_dir] + 180 # wind direction is 180 deg offset from wind vector
        data[col_spd] = np.sqrt(data[U]**2 + data[V]**2)    
        
    return data



def read_slow_data(datafile):
    
    ''' 
    - reads data from inflow mast PTU and cup anemometer (1Hz data)
    '''
    
    # print (datafile)
    
    # read data
    # skip = [0]
    skip = [0,2,3]
    
    data = pd.read_csv(datafile,
                    index_col = 0,
                    header = 0,    
                    skiprows = skip, 
                    engine = 'c',
                    on_bad_lines='warn', 
                    na_values = {'NAN'},
                    dtype = float,
                    parse_dates=True
                        )

    # if 'RECNBR' in data:        
    #     data.drop(['RECNBR'], axis=1, inplace=True)

    return data


def resample_sonic(data, resample_freq):
    
    ''' 
    - resample datafile
    - calculate wind direction and horizontal wind again after resampling
    '''
    
    # resample
    #data = data.astype(float)
    data = data.select_dtypes('number').resample(resample_freq).mean()
    
    # calculate wind direction and horizontal wind again after resampling (mean of wind direction does not work!)
    data = calc_wind(data)


    return data


#%% Other functions

def sun_elev_to_trough_angles(elev_angles, azimuth_angles):
    # trough_angles = np.degrees( np.arctan2(np.sin(np.radians(elev_angles)), np.sin(np.radians(azimuth_angles)) ))
    # print('trough angle = {:2f}'.format(trough_angles))
    # # print(trough_angles)
    # # trough_angles = trough_angles.where(trough_angles.isnull()==False, -30)
    # # print(trough_angles.where(trough_angles.isnull()==False, -30))
    # trough_angles = -trough_angles + 90
    # print('trough angle = {:2f}'.format(trough_angles))
    x, _, z = get_aimpt_from_sunangles(elev_angles, azimuth_angles)
    trough_angle = get_tracker_angle_from_aimpt(x,z)
    return trough_angle

def get_aimpt_from_sunangles(elev_angles, azimuth_angles):
    # trough_angles = sun_elev_to_trough_angles(elev_angles, azimuth_angles)
    # print('elev angle = {:2f}'.format(elev_angles))
    # print('azimuth angle = {:2f}'.format(azimuth_angles))
    # #print('trough angle = {:2f}'.format(trough_angles))
    # signed_elev_angles = 90 - trough_angles
    # x = factor * np.cos(np.radians(signed_elev_angles))
    # z = x * np.tan(np.radians(signed_elev_angles))
    x = np.cos(np.radians(elev_angles))*np.sin(np.radians(azimuth_angles))
    y = np.cos(np.radians(elev_angles)) * np.cos(np.radians(azimuth_angles))
    z = np.sin(np.radians(elev_angles))
    return x,y,z

def get_tracker_angle_from_aimpt(x,z):
    tracker_angle = np.degrees(np.arctan2(x,z))
    return tracker_angle


def create_file_structure(file_path, resolution, year, month, day):
    try:
         # Create the directory structure (resolution not needed anymore)
        if type(month)==str:
            directory = os.path.join(file_path, "year="+year+"/month="+month+"/day="+day)
        else:
            directory = os.path.join(file_path, f"year={year:04d}/month={month:02d}/day={day:02d}")

        # If the directory already exists, delete all files within it
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)

        # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        return directory

    except Exception as e:
        print(f"Error: {e}")
                


#%%
if __name__ == "__main__":
    
    pass




