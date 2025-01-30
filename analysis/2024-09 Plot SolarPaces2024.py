import pandas as pd
import s3fs
import numpy as np
import pyarrow
from windrose import WindroseAxes
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates
import scipy as sp
import scipy.signal
import sys
import time
import glob
import pickle
import datetime
import matplotlib.cm as cm
from matplotlib import rcParams, rcParamsDefault
import pytz


plt.rc('font', size=14) 



sys.path.append("C:/Users/uegerer/Desktop/NSO/NSO_data_processing")
sys.path.append("C:/Users/uegerer/Desktop/NSO/NSO_analysis")
sys.path.append("C:/Users/uegerer/Desktop/NSO/NSO_loads_processing")


from Functions_general import *
from Functions_loads import *
from Functions_masts_CD import *





#%% Definitions and read data



# Define date to read data (can be multiple options with '*')
year = '*'
month = '*'
day = '*'


# Read the datasets for all dates

years   = [2024] 
months  = [7,8,9] # 
days    = np.arange(1,31)   # [15,16,17]   # 




path = 'CD_processed_data_preliminary/'    #  'Y:\Wind-data/Restricted/Projects/NSO/CrescentDunes_processed_met_tower_data_preliminary/'  #  


inflow_files = []
mast_files = []
for year in years:
    year = str(year)

    for month in months:
        
        if year == '2024' and month<3: # no data before March 2024
            continue
        
        month = f'{month:02d}'
    
        for day in days:
            day = f'{day:02d}'
              
            inflow_files = inflow_files + sorted(glob.glob(path +'Inflow_Mast_1min_' + year + '-' + month + '-' + day + '_' + '*.pkl'))   #
            mast_files = mast_files + sorted(glob.glob(path +'Wake_masts_1min_' + year + '-' + month + '-' + day + '_' + '*.pkl'))   #        



### Read data
inflow = pd.DataFrame()    
for datafile in inflow_files:
    inflow = pd.concat( [inflow, pd.read_pickle(datafile)]) 
inflow = inflow.sort_index()

# cut inflow, so that only data with sonics is used
inflow = inflow[:]

masts = pd.DataFrame()
for datafile in mast_files:
    masts = pd.concat( [masts, pd.read_pickle(datafile)] )   
masts = masts.sort_index()



# inflow.to_parquet("1min_data/inflow.parquet")
# masts.to_parquet("1min_data/masts.parquet")
# loads.to_parquet("1min_data/loads.parquet")


# inflow = pd.read_parquet("1min_data/inflow.parquet")
# masts = pd.read_parquet("1min_data/masts.parquet")
# loads = pd.read_parquet("1min_data/loads.parquet")


# inflow = inflow[loads.index[0]:loads.index[-1]]
# masts  = masts[loads.index[0]:loads.index[-1]]

inflow['rho']    = rho(inflow.p, inflow.RH, inflow.Temp).interpolate(limit_direction='both')
#inflow['u_star'] = (abs(inflow.Tau.interpolate()) / inflow.rho)**0.5 


### Filter wind directions

# Use only wind direction when wind speed greater than defined wind speed limit
wind_dir_limit = 0.0 # m/s
for col_dir in [col for col in inflow.columns if 'wdir_' in col]:    # loop over every inflow mast height      
    col_spd = col_dir.replace("wdir", "wspd") # column name for wind speed
    inflow[col_dir] = inflow[col_dir].where(inflow[col_spd] > wind_dir_limit)
for col_dir in [col for col in masts.columns if 'wdir_' in col]:    # loop over every inflow mast height      
    col_spd = col_dir.replace("wdir", "wspd") # column name for wind speed
    masts[col_dir] = masts[col_dir].where(masts[col_spd] > wind_dir_limit)
    
   

# Convert time zone to local
pacific_tz = pytz.timezone('America/Los_Angeles')

# Assuming 'index' of DataFrames is already in UTC
inflow.index = inflow.index.tz_localize('UTC').tz_convert(pacific_tz)
masts.index = masts.index.tz_localize('UTC').tz_convert(pacific_tz)


# SCADA data
  
def read_SCADA_file(paths, filename_search_strings):
    
    
    # filename = "Report_14_MAR_2024*.csv"
    
    # path = '../SCADA/EMS1/'  
    
    # Initialize an empty list to store the file paths
    all_files = []
    
    # Loop through each path and filename search string
    for path in paths:
        for filename_search_string in filename_search_strings:
            files = glob.glob(os.path.join(path, filename_search_string))
            all_files.extend(files)  # Add files to the list
        
    files = all_files # glob.glob(os.path.join(path, filename_search_string))
    
    
    df_list = []

    # Loop through the files and read each one into a dataframe
    for file in files:
        df = pd.read_csv(file, skiprows=[2], header=[0, 1], index_col=0, 
                         parse_dates=True, date_format='%d-%b-%Y %H:%M:%S.%f PDT ', 
                         low_memory=False)
        df_list.append(df)
    
    # Concatenate all dataframes into one
    df = pd.concat(df_list)
        
    # Convert the index from PDT to UTC
    df.index = df.index.tz_localize('America/Los_Angeles').tz_convert('UTC')
    
    df = df.sort_index()
    
    
    # Clean up the column names by stripping leading and trailing spaces
    df.columns = df.columns.map(lambda x: (x[0].strip(), x[1].strip()))
    
    
            
    
    # Drop columns that have "Hex" in the second row
    df = df.loc[:, df.columns.get_level_values(1) != '(Hex)']
    
    # Flatten the multi-level columns - or drop second header column
    #df.columns = df.columns.map(' '.join).str.strip()
    df.columns = df.columns.get_level_values(0)
    
    


    H1 = df[df['Heliostat'] == 'W2-74-11'].drop(columns=["Heliostat"])
    H2 = df[df['Heliostat'] == 'W2-73-25'].drop(columns=["Heliostat"])
    H3 = df[df['Heliostat'] == 'W2-58-16'].drop(columns=["Heliostat"])
    
    for H in [H1, H2, H3]:
        H = H1.apply(pd.to_numeric, errors='coerce')
        H = H.resample("10min").mean()
        
        
    return H1, H2, H3


H1, H2, H3 = read_SCADA_file(paths = ['../SCADA/EMS1/', '../SCADA/EMS2/'], 
                             filename_search_strings = ["Report*_SEP_2024*.csv"])




H1 = H1.resample("min").first()   # .tz_localize(None)
inflow = pd.merge(inflow, H1[['State', 'AngAzData', 'AngElData']], left_index=True, right_index=True, suffixes=('', '_H1'), how="outer")
inflow["side_angle"] = (inflow["wdir_Mid"] - inflow["AngAzData"] ) % 360 - 180
inflow.AngElData = (inflow.AngElData - 90 ) * -1


inflow.index = inflow.index.tz_convert(pacific_tz)

inflow = inflow[masts.index[0]:]


plt.rcParams.update({'font.size': 19.5})

#%% Overview

Overview = 1
if Overview == 1:
    
    
    
    # # Read metar data
    # met = read_metar(station='TPH', 
    #                        start_year=inflow.index[0].year, start_month=inflow.index[0].month, start_day=inflow.index[0].day, 
    #                        end_year=inflow.index[-1].year, end_month=inflow.index[-1].month, end_day=inflow.index[-1].day)
    # met = met[inflow.index[0]: inflow.index[-1]]
    
    height = "Top"

    
    # Timeseries of wind
    fig = plt.figure(figsize=(16,9.5))
    
    ms = 2 
    
    ax2 = plt.subplot(4, 1, 1)
    plt.ylabel('Wind speed (m/s)')
    for height_col in [col for col in inflow.columns if ('wspd_' + height in col) & ("_m" not in col) & ("_s" not in col)]:     
        ax2.plot(inflow[height_col],'s', label = "inflow "+height_col[-3:], color='black', ms=ms)
    for height_col in [col for col in masts.columns if ('m1_wspd_' + height in col) & ("_m" not in col) & ("_s" not in col)]:     
        ax2.plot(masts[height_col],'o', label =  "mast 1 "+height_col[-3:], color='orange', ms=ms)
    for height_col in [col for col in masts.columns if ('m3_wspd_' + height in col) & ("_m" not in col) & ("_s" not in col)]:      
        ax2.plot(masts[height_col],'>', label =  "mast 3 "+height_col[-3:], color='blue', ms=ms)
    # plt.plot(met.sknt/1.94384,"o", color='grey', label = "METAR", ms=ms)  
    plt.legend(loc=1,  markerscale=8/ms)
    plt.grid()
    ax2.set_zorder(100)
    
    ax1 = plt.subplot(4, 1, 2, sharex=ax2)  # 
    ax1.set_ylabel('Wind dir ($^\circ$)  ')
    for height_col in [col for col in inflow.columns if 'wdir_' + height in col]:     
        ax1.plot(inflow[height_col],'s', label = "", color='black', ms=ms)
    for height_col in [col for col in masts.columns if ('m1_wdir_' + height in col) & ("_m" not in col) & ("_s" not in col)]:    # loop over every mast height   
        ax1.plot(masts[height_col],'o', label = height_col, color='orange', ms=ms)
    for height_col in [col for col in masts.columns if ('m3_wdir_' + height in col) & ("_m" not in col) & ("_s" not in col)]: 
        ax1.plot(masts[height_col],'>', label = height_col, color='blue', ms=ms)
    # plt.plot(met.drct,"o", color='grey', label = "METAR", ms=ms) 
    ax1.set_ylim(0, 360)    
    plt.grid() 
    
    ax1 = plt.subplot(4, 1, 3, sharex=ax2)  # 
    ax1.set_ylabel('TKE (m$^{2}$/s$^{2}$)')
    for height_col in [col for col in inflow.columns if 'TKE_' + height in col]:     
        ax1.plot(inflow[height_col],'s', label = "", color='black', ms=ms)
    for height_col in [col for col in masts.columns if ('m1_TKE_' + height in col) & ("_m" not in col) & ("_s" not in col)]:    # loop over every mast height   
        ax1.plot(masts[height_col],'o', label = height_col, color='orange', ms=ms)
    for height_col in [col for col in masts.columns if ('m3_TKE_' + height in col) & ("_m" not in col) & ("_s" not in col)]: 
        ax1.plot(masts[height_col],'>', label = height_col, color='blue', ms=ms)
    plt.grid() 
    
    ax1 = plt.subplot(4, 1, 4, sharex=ax2)  # 
    ax1.set_ylabel('Angle ($^\circ$)')  
    plt.plot( inflow.AngAzData,"o",  ms=ms, label = "H1 Azimuth")
    plt.plot( inflow.AngElData,"o",  ms=ms, label = "H1 Elevation")
    #plt.plot( inflow.side_angle,".",  ms=ms, label = "H1 side angle")
    plt.legend(markerscale=8/ms, loc="upper right")
    plt.grid()  

    fig.autofmt_xdate()
    ax1.set_xlabel("Date & Time (local)")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    
    #fig.savefig('C:/Users/uegerer/Desktop/Overview_time_series.pdf', dpi=300)



#%% wind deficit

wind_deficit = 0 
if wind_deficit == 1: 
                    
    
    all = inflow.merge(masts, left_index=True, right_index=True, how="inner")
                    
    all["wind_reduction"] = (all.wspd_Mid - all.m3_wspd_Mid)/all.wspd_Top
  


    plt.figure()
    plt.xlabel("Wind dir ($^\circ$)")
    plt.xlabel("Wind reduction (m/s)")
    #plt.plot(all.wdir_Top, all.wind_reduction, ".", color="black")
    plt.scatter(all.wdir_Top, all.wind_reduction, c=all.AngElData, cmap="viridis")
    plt.colorbar(label="Azimuth")





    









#%% Inflow wind profile
inflow_wind_profile = 0
if inflow_wind_profile == 1:
    
        ymin = 0
        ymax = 16
        
        # example day
        ex = inflow["2024-03-14" : "2024-03-14"].median()

         
        c11 = "red"
        c12 = "blue"
        c13 = "green"
        
        c1 = 'black'
    
        fig = plt.subplots( figsize=(12.0, 7.0) ) 
        plt.axis('off')
        
        ## Wind speed
        ax0 = plt.subplot(1,3, 1)
        ax0.set_xlabel("Wind speed (m/s)")
        heights =      [15,11,5.5,2.75]   
        winds =        ['WS_15m','wspd_Top', 'wspd_Mid', 'wspd_Low' ]
        winds_mean = []
        for (height, wind) in zip(heights, winds):
            #print (height, wind)
            winds_mean = winds_mean + [inflow[wind].mean()]
            box = plt.boxplot(inflow[wind].dropna(), positions=[height], showfliers=False, vert=False, widths = 1, patch_artist=True, showmeans=True, 
                    boxprops=dict(fill=None, color=c1),
                    capprops=dict(color=c1),
                    whiskerprops=dict(color=c1),
                    medianprops=dict(color=c1),
                    meanprops={'markerfacecolor': c1, 'markeredgecolor': c1}) 
        plt.plot( [ex.WS_15m,ex.wspd_Top, ex.wspd_Mid, ex.wspd_Low ], [15,11,5.5,2.75])
        plt.grid()  
        ax0.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        # plt.xlabel('Wind direction inflow (m/s)')
        #plt.xlim(0,13)
        plt.ylim(ymin, ymax)     
        plt.ylabel("Height (m)")
        
        
        ## Wind direction
        ax0 = plt.subplot(1,3, 2)
        ax0.set_xlabel("Wind dir ($^\circ$)")       
        heights =      [11,5.5,2.75]   
        winds =        ['TI_w_Top', 'TI_w_Mid', 'TI_w_Low' ]
        winds_mean = []
        for (height, wind) in zip(heights, winds):
            #print (height, wind)
            winds_mean = winds_mean + [inflow[wind].median()]
            box = plt.boxplot(inflow[wind].dropna(), positions=[height], showfliers=False, vert=False, widths = 1, patch_artist=True, showmeans=False, 
                    boxprops=dict(fill=None, color=c1),
                    capprops=dict(color=c1),
                    whiskerprops=dict(color=c1),
                    medianprops=dict(color=c1),
                    #meanprops={'markerfacecolor': c1, 'markeredgecolor': c1}
                    ) 
        plt.plot( [ex.TI_w_Top, ex.TI_w_Mid, ex.TI_w_Low ], [11,5.5,2.75])
        plt.grid()  
        ax0.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        # plt.xlabel('Wind direction inflow (m/s)')
        #plt.xlim(0,13)
        plt.ylim(ymin, ymax)    
        ax0.set_yticklabels([]) 
        

        
        
        ## TKE
        ax0 = plt.subplot(1,3, 3)
        ax0.set_xlabel("TKE (m$^2$/s$^2$)")   
        winds =        ['TKE_Top', 'TKE_Mid', 'TKE_Low' ]
        for (height, wind) in zip(heights, winds):
            #print (height, wind)
            plt.boxplot(inflow[wind].dropna(), positions=[height], showfliers=False, vert=False, widths = 1, patch_artist=True, showmeans=True,
                    boxprops=dict(fill=None, color=c1),
                    capprops=dict(color=c1),
                    whiskerprops=dict(color=c1),
                    medianprops=dict(color=c1),
                    meanprops={'markerfacecolor': c1, 'markeredgecolor': c1})  
        
       # plt.plot( [ex.TKE_Top, ex.TKE_Mid, ex.TKE_Low ], [11,5.5,2.75], label = ex.index.date() )
        plt.grid()  
        ax0.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        plt.ylim(ymin, ymax) 
        plt.xlim(0,6.6)
        ax0.set_yticklabels([]) 
        ax0.legend()
        
        plt.suptitle("Average wind conditions {} to {}".format(inflow.index[0].date(), inflow.index[-1].date()))
        
        plt.tight_layout()
       # plt.subplots_adjust( wspace=0.1, hspace=0.4)

 
    
 



#%% Wind roses for all masts    


Winds_over_troughs = 1
if Winds_over_troughs == 1:
    

    inflow_west_with_masts =  inflow#_west.loc[inflow_west.index.intersection(masts_west.index)]
    
    masts_west = masts
    
    fontsize = 21
    

    
    # fig = plt.figure(figsize=(13,8))
    # bins = np.arange(0, 14, 2)
    
    # fig.suptitle("Inflow mast, 11m, at CD from {} to {}".format(inflow.index[0].date(), inflow.index[-1].date()))
    # ax = fig.add_subplot(1, 1, 1, projection="windrose")
    # ws = inflow_west_with_masts.dropna(subset=['wspd_Top', 'wdir_Top']).wspd_Top
    # wd = inflow_west_with_masts.dropna(subset=['wspd_Top', 'wdir_Top']).wdir_Top
    # # ws = met.dropna(subset=['sknt', 'drct']).sknt/1.94384
    # # wd = met.dropna(subset=['sknt', 'drct']).drct
    # ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
    # ax.set_legend(title="Wind speed (m/s)", loc = "lower right")   
    
        

    nrows, ncols = 3, 4
    fig = plt.figure(figsize=(15,9))
    bins = np.arange(0, 11, 1.5)
    
    # fig.suptitle("Masts from {} to {}, only ".format(inflow.index[0].date(), inflow.index[-1].date())   + " winds")
    # fig.suptitle("Wind flow modification over heliostats")
    
    ax = fig.add_subplot(nrows, ncols, 1, projection="windrose")
    ws = inflow_west_with_masts.dropna(subset=['wspd_Top', 'wdir_Top']).wspd_Top
    wd = inflow_west_with_masts.dropna(subset=['wspd_Top', 'wdir_Top']).wdir_Top
    ax.set_title("Inflow mast, 11m ".format(len(ws)))  # \n {} data points
    ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
    ax.set_yticklabels([])
    
    ax = fig.add_subplot(nrows, ncols, 5, projection="windrose")
    ws = inflow_west_with_masts.dropna(subset=['wspd_Mid', 'wdir_Mid']).wspd_Mid
    wd = inflow_west_with_masts.dropna(subset=['wspd_Mid', 'wdir_Mid']).wdir_Mid
    ax.set_title("Inflow mast, 5.5m".format(len(ws)))
    ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
    ax.set_yticklabels([])
    #ax.set_legend(title="Wind speed (m/s)", loc = "lower right")   
    
    ax = fig.add_subplot(nrows, ncols, 9, projection="windrose")
    ws = inflow_west_with_masts.dropna(subset=['wspd_Low', 'wdir_Low']).wspd_Low
    wd = inflow_west_with_masts.dropna(subset=['wspd_Low', 'wdir_Low']).wdir_Low
    ax.set_title("Inflow mast, 2.75m".format(len(ws)))
    ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
    ax.set_yticklabels([])
        
    ax = fig.add_subplot(nrows, ncols, 2, projection="windrose")
    ws = masts_west.dropna(subset=['m1_wspd_Top', 'm1_wdir_Top']).m1_wspd_Top
    wd = masts_west.dropna(subset=['m1_wspd_Top', 'm1_wdir_Top']).m1_wdir_Top
    ax.set_title("Wake mast 1, 11m".format(len(ws)))
    ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60) 
    ax.set_yticklabels([])
    
    ax = fig.add_subplot(nrows, ncols, 6, projection="windrose")
    ws = masts_west.dropna(subset=['m1_wspd_Mid', 'm1_wdir_Mid']).m1_wspd_Mid
    wd = masts_west.dropna(subset=['m1_wspd_Mid', 'm1_wdir_Mid']).m1_wdir_Mid
    ax.set_title("Wake mast 1, 5.5m".format(len(ws)))
    ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
    ax.set_yticklabels([])
    
    # ax = fig.add_subplot(nrows, ncols, 10, projection="windrose")
    # ws = masts_west.dropna(subset=['m1_wspd_Low', 'm1_wdir_Low']).m1_wspd_Low
    # wd = masts_west.dropna(subset=['m1_wspd_Low', 'm1_wdir_Low']).m1_wdir_Low
    # ax.set_title("Wake mast 1, 5.5m".format(len(ws)))
    # ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
    # ax.set_yticklabels([])
    
    ax = fig.add_subplot(nrows, ncols, 3, projection="windrose")
    ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
    l=ax.set_legend(title="  Wind speed (m/s)  ", loc = "center", framealpha=1, fontsize = fontsize )
    plt.setp(l.get_texts(), fontsize=fontsize)
    
    # ax = fig.add_subplot(nrows, ncols, 3, projection="windrose")
    # ws = masts_west.dropna(subset=['m2_wspd_Top', 'm2_wdir_Top']).m2_wspd_Top
    # wd = masts_west.dropna(subset=['m2_wspd_Top', 'm2_wdir_Top']).m2_wdir_Top
    # ax.set_title("Wake mast 2, 11m".format(len(ws)))
    # if len (ws)>0:
    #     ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)  
    # ax.set_yticklabels([])
    
    # ax = fig.add_subplot(nrows, ncols, 7, projection="windrose")
    # ws = masts_west.dropna(subset=['m2_wspd_Mid', 'm2_wdir_Mid']).m2_wspd_Mid
    # wd = masts_west.dropna(subset=['m2_wspd_Mid', 'm2_wdir_Mid']).m2_wdir_Mid
    # ax.set_title("Wake mast 2, 5.5m".format(len(ws)))
    # if len (ws)>0:
    #     ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
    # ax.set_yticklabels([])
    
    # ax = fig.add_subplot(nrows, ncols, 11, projection="windrose")
    # ws = masts_west.dropna(subset=['m2_wspd_Low', 'm2_wdir_Low']).m2_wspd_Low
    # wd = masts_west.dropna(subset=['m2_wspd_Low', 'm2_wdir_Low']).m2_wdir_Low
    # ax.set_title("Wake mast 2, 2.75m".format(len(ws)))
    # if len (ws)>0:
    #     ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
    # ax.set_yticklabels([])
    
    ax = fig.add_subplot(nrows, ncols, 4, projection="windrose")
    ws = masts_west.dropna(subset=['m3_wspd_Top', 'm3_wdir_Top']).m3_wspd_Top
    wd = masts_west.dropna(subset=['m3_wspd_Top', 'm3_wdir_Top']).m3_wdir_Top
    ax.set_title("Wake mast 3, 11m".format(len(ws)))
    if len (ws)>0:
        ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)   
        ax.set_yticklabels([])
    
    ax = fig.add_subplot(nrows, ncols, 8, projection="windrose")
    ws = masts_west.dropna(subset=['m3_wspd_Mid', 'm3_wdir_Mid']).m3_wspd_Mid
    wd = masts_west.dropna(subset=['m3_wspd_Mid', 'm3_wdir_Mid']).m3_wdir_Mid
    ax.set_title("Wake mast 3, 5.5m".format(len(ws)))
    if len (ws)>0:
        ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
        ax.set_yticklabels([])
        
    ax = fig.add_subplot(nrows, ncols, 12, projection="windrose")
    ws = masts_west.dropna(subset=['m3_wspd_Low', 'm3_wdir_Low']).m3_wspd_Low
    wd = masts_west.dropna(subset=['m3_wspd_Low', 'm3_wdir_Low']).m3_wdir_Low
    ax.set_title("Wake mast 3, 2.75m".format(len(ws)))
    if len (ws)>0:
        ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
        ax.set_yticklabels([])
             
    plt.tight_layout()
    
    plt.subplots_adjust( wspace=0.4, hspace=0.7)
    
    #fig.savefig('wind_paper_plots/Wind_over_troughs.png', dpi=300) 
    
    
    #%% Wind shear
    
        
    inflow_west_with_masts["wind_shear"] = inflow_west_with_masts.wspd_Top - inflow_west_with_masts.wspd_Mid
    
    masts_west["m1_wind_shear"] = masts_west.m1_wspd_Top - masts_west.m1_wspd_Mid
   # masts_west["m2_wind_shear"] = masts_west.m2_wspd_Top - masts_west.m2_wspd_Mid
    masts_west["m3_wind_shear"] = masts_west.m3_wspd_Top - masts_west.m3_wspd_Mid

    nrows, ncols = 1, 4
    fig = plt.figure(figsize=(16,5))
    bins = np.arange(-0.5,4, 0.5)
    
    # fig.suptitle("Masts from {} to {}, only ".format(inflow.index[0].date(), inflow.index[-1].date())   + " winds")
    # fig.suptitle("Wind flow modification over heliostats")
    
    ax = fig.add_subplot(nrows, ncols, 1, projection="windrose")
    ws = inflow_west_with_masts.dropna(subset=['wind_shear', 'wdir_Top']).wind_shear
    wd = inflow_west_with_masts.dropna(subset=['wind_shear', 'wdir_Top']).wdir_Top
    ax.set_title("Inflow mast".format(len(ws)))  # \n {} data points
    ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
    ax.set_yticklabels([])
    
         
    ax = fig.add_subplot(nrows, ncols, 2, projection="windrose")
    ws = masts_west.dropna(subset=['m1_wind_shear', 'm1_wdir_Top']).m1_wind_shear
    wd = masts_west.dropna(subset=['m1_wind_shear', 'm1_wdir_Top']).m1_wdir_Top
    ax.set_title("Wake mast 1".format(len(ws)))
    ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60) 
    ax.set_yticklabels([])


    # ax = fig.add_subplot(nrows, ncols, 3, projection="windrose")
    # ws = masts_west.dropna(subset=['m2_wind_shear', 'm2_wdir_Top']).m2_wind_shear
    # wd = masts_west.dropna(subset=['m2_wind_shear', 'm2_wdir_Top']).m2_wdir_Top
    # ax.set_title("Wake mast 2, 11m".format(len(ws)))
    # if len (ws)>0:
    #     ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)  
    # ax.set_yticklabels([])
        
    ax = fig.add_subplot(nrows, ncols, 3, projection="windrose")
    ws = masts_west.dropna(subset=['m3_wind_shear', 'm3_wdir_Top']).m3_wind_shear
    wd = masts_west.dropna(subset=['m3_wind_shear', 'm3_wdir_Top']).m3_wdir_Top
    ax.set_title("Wake mast 3")
    if len (ws)>0:
        ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)   
        ax.set_yticklabels([])
        
    ax = fig.add_subplot(nrows, ncols, 4, projection="windrose")
    ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
    l=ax.set_legend(title="Wind shear, 11m-5.5m (m/s)", loc = "center", framealpha=1)
    plt.setp(l.get_texts(), fontsize=fontsize)
    
    
             
    plt.tight_layout()
    

    
    
     
#%%

    nrows, ncols = 3, 4
    fig = plt.figure(figsize=(15,9))
    bins = np.logspace(-1,0.5,7)
    
    # fig.suptitle("Masts from {} to {}, only ".format(inflow.index[0].date(), inflow.index[-1].date())   + " winds")
    # fig.suptitle("Wind flow modification over heliostats")
    
    ax = fig.add_subplot(nrows, ncols, 1, projection="windrose")
    ws = inflow_west_with_masts.dropna(subset=['TKE_Top', 'wdir_Top']).TKE_Top
    wd = inflow_west_with_masts.dropna(subset=['TKE_Top', 'wdir_Top']).wdir_Top
    ax.set_title("Inflow mast, 11m ".format(len(ws)))  # \n {} data points
    ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
    ax.set_yticklabels([])
    
    ax = fig.add_subplot(nrows, ncols, 5, projection="windrose")
    ws = inflow_west_with_masts.dropna(subset=['TKE_Mid', 'wdir_Mid']).TKE_Mid
    wd = inflow_west_with_masts.dropna(subset=['TKE_Mid', 'wdir_Mid']).wdir_Mid
    ax.set_title("Inflow mast, 5.5m".format(len(ws)))
    ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
    ax.set_yticklabels([])
    #ax.set_legend(title="Wind speed (m/s)", loc = "lower right")   
    
    ax = fig.add_subplot(nrows, ncols, 9, projection="windrose")
    ws = inflow_west_with_masts.dropna(subset=['TKE_Low', 'wdir_Low']).TKE_Low
    wd = inflow_west_with_masts.dropna(subset=['TKE_Low', 'wdir_Low']).wdir_Low
    ax.set_title("Inflow mast, 2.75m".format(len(ws)))
    ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
    ax.set_yticklabels([])
        
    ax = fig.add_subplot(nrows, ncols, 2, projection="windrose")
    ws = masts_west.dropna(subset=['m1_TKE_Top', 'm1_wdir_Top']).m1_TKE_Top
    wd = masts_west.dropna(subset=['m1_TKE_Top', 'm1_wdir_Top']).m1_wdir_Top
    ax.set_title("Wake mast 1, 11m".format(len(ws)))
    ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60) 
    ax.set_yticklabels([])
    
    ax = fig.add_subplot(nrows, ncols, 6, projection="windrose")
    ws = masts_west.dropna(subset=['m1_TKE_Mid', 'm1_wdir_Mid']).m1_TKE_Mid
    wd = masts_west.dropna(subset=['m1_TKE_Mid', 'm1_wdir_Mid']).m1_wdir_Mid
    ax.set_title("Wake mast 1, 5.5m".format(len(ws)))
    ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
    ax.set_yticklabels([])
    
    # ax = fig.add_subplot(nrows, ncols, 10, projection="windrose")
    # ws = masts_west.dropna(subset=['m1_TKE_Low', 'm1_wdir_Low']).m1_TKE_Low
    # wd = masts_west.dropna(subset=['m1_TKE_Low', 'm1_wdir_Low']).m1_wdir_Low
    # ax.set_title("Wake mast 1, 5.5m".format(len(ws)))
    # ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
    # ax.set_yticklabels([])
    
    ax = fig.add_subplot(nrows, ncols, 3, projection="windrose")
    ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
    l=ax.set_legend(title="    TKE (m$^2$/s$^2$)    ", loc = "center", framealpha=1)
    plt.setp(l.get_texts(), fontsize=fontsize)
    
    # ax = fig.add_subplot(nrows, ncols, 3, projection="windrose")
    # ws = masts_west.dropna(subset=['m2_TKE_Top', 'm2_wdir_Top']).m2_TKE_Top
    # wd = masts_west.dropna(subset=['m2_TKE_Top', 'm2_wdir_Top']).m2_wdir_Top
    # ax.set_title("Wake mast 2, 11m".format(len(ws)))
    # if len (ws)>0:
    #     ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)  
    # ax.set_yticklabels([])
    
    # ax = fig.add_subplot(nrows, ncols, 7, projection="windrose")
    # ws = masts_west.dropna(subset=['m2_TKE_Mid', 'm2_wdir_Mid']).m2_TKE_Mid
    # wd = masts_west.dropna(subset=['m2_TKE_Mid', 'm2_wdir_Mid']).m2_wdir_Mid
    # ax.set_title("Wake mast 2, 5.5m".format(len(ws)))
    # if len (ws)>0:
    #     ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
    # ax.set_yticklabels([])
    
    # ax = fig.add_subplot(nrows, ncols, 11, projection="windrose")
    # ws = masts_west.dropna(subset=['m2_TKE_Low', 'm2_wdir_Low']).m2_TKE_Low
    # wd = masts_west.dropna(subset=['m2_TKE_Low', 'm2_wdir_Low']).m2_wdir_Low
    # ax.set_title("Wake mast 2, 2.75m".format(len(ws)))
    # if len (ws)>0:
    #     ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
    # ax.set_yticklabels([])
    
    ax = fig.add_subplot(nrows, ncols, 4, projection="windrose")
    ws = masts_west.dropna(subset=['m3_TKE_Top', 'm3_wdir_Top']).m3_TKE_Top
    wd = masts_west.dropna(subset=['m3_TKE_Top', 'm3_wdir_Top']).m3_wdir_Top
    ax.set_title("Wake mast 3, 11m".format(len(ws)))
    if len (ws)>0:
        ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)   
        ax.set_yticklabels([])
    
    ax = fig.add_subplot(nrows, ncols, 8, projection="windrose")
    ws = masts_west.dropna(subset=['m3_TKE_Mid', 'm3_wdir_Mid']).m3_TKE_Mid
    wd = masts_west.dropna(subset=['m3_TKE_Mid', 'm3_wdir_Mid']).m3_wdir_Mid
    ax.set_title("Wake mast 3, 5.5m".format(len(ws)))
    if len (ws)>0:
        ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
        ax.set_yticklabels([])
        
    ax = fig.add_subplot(nrows, ncols, 12, projection="windrose")
    ws = masts_west.dropna(subset=['m3_TKE_Low', 'm3_wdir_Low']).m3_TKE_Low
    wd = masts_west.dropna(subset=['m3_TKE_Low', 'm3_wdir_Low']).m3_wdir_Low
    ax.set_title("Wake mast 3, 2.75m".format(len(ws)))
    if len (ws)>0:
        ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
        ax.set_yticklabels([])
             
    plt.tight_layout()
    plt.subplots_adjust( wspace=0.4, hspace=0.7)
    
        
