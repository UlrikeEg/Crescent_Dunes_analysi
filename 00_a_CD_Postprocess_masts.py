import numpy as np
from numpy import cos,sin
import pandas as pd
import matplotlib as mpl
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

sys.path.append("C:/Users/uegerer/Desktop/NSO/NSO_data_processing")

from Functions_masts_CD import *
from Functions_general import *

mpl.rcParams['lines.markersize'] = 2


years   =  np.arange(2024,2025)   
months  =  np.arange(1,13)   
days    =  np.arange(1,32)   

years   =  [2024] #
months  =  [3,4] # 
days    =  np.arange(1,32)  # [16] # 

start_processing = pd.to_datetime('2024-03-12 00:00:00')
#start_processing = pd.to_datetime('2024-04-14 00:00:00')


fs = 20 # [Hz], sampling frequency

inflow_path = 'Y:\Wind-data/Restricted/Projects/NSO/CrescentDunes_Met_Inflow_Sonics/'
inflow_slow_path = 'Y:\Wind-data/Restricted/Projects/NSO/CrescentDunes_Met_Inflow_Low_Speed_Data/'
mast1_path = 'Y:\Wind-data/Restricted/Projects/NSO/xxx/'
mast2_path = 'Y:\Wind-data/Restricted/Projects/NSO/xxx/'
mast3_path = 'Y:\Wind-data/Restricted/Projects/NSO/xxx/'

path_save = './CD_processed_data_preliminary/' # 'Y:\Wind-data/Restricted/Projects/NSO/CrescentDunes_processed_met_tower_data_preliminary/'  # 
            
            



debug = 0
save  = 1
plot  = 0



               


for year in years:
    year = str(year)

    for month in months:
               
        month = f'{month:02d}'
    
        for day in days:
            day = f'{day:02d}'
              
            print (year, month, day)   
            
            try:
                if pd.to_datetime(year+ month+ day) <= start_processing: 
                    continue

            except:
                pass
            
            #%% Read data
        
            
            ### Read inflow mast sonic files  
            
            inflow_files = (sorted(glob.glob(inflow_path +
                                             'CD_Inflow_Sonics_' + year + '_' + month + '_' + day + '_' + '*.dat'))  ) 
            
            # try:
            #     if pd.to_datetime(year+ month+ day) == pd.to_datetime('2024-03-16 00:00:00'):
            #         inflow_files = ['Y:\\Wind-data/Restricted/Projects/NSO/CrescentDunes_Met_Inflow_Sonics\\CD_Inflow_Sonics_2024_03_15_0548.dat'] + inflow_files   # this file contains data for several hours, including March 16th
            # except:
            #     pass
            
            
            # Add all files on the day to a single Dataframe
            inflow = pd.DataFrame()
            for datafile in inflow_files:
                inflow = pd.concat( [inflow, read_sonic(datafile)]) 
            inflow = inflow.drop_duplicates().sort_index()
        
            ### Read inflow mast slow files     
        
            # inflow_slow_files = ( sorted(glob.glob(inflow_slow_path +
            #                                        'Inflow_Met_Mast_LowSpeedData_' + year + '_' + month + '_' + day + '_' + '*.dat')) )
            
            inflow_slow_files = ( sorted(glob.glob(inflow_slow_path +
                                                   'TOA5_21544.LowSpeedData_' + year + '_' + month + '_' + day + '_' + '*.dat')) )
            
            # Check if DataFrame spans more than a day and if yes append slow inflow files
            if len(inflow)!=0:
                if (inflow.index[-1] - inflow.index[0]).days >= 1:
                    for i in range(1,(inflow.index[-1] - inflow.index[0]).days):
                        day_after = pd.to_datetime(year + month + day).date()+ pd.to_timedelta(i,"D")
                        inflow_slow_files = inflow_slow_files+(
                                             sorted(glob.glob(inflow_slow_path +
                                                              #'Inflow_Met_Mast_LowSpeedData_' 
                                                              'TOA5_21544.LowSpeedData_'
                                                              + day_after.strftime('%Y') + '_' + day_after.strftime('%m') + '_' + 
                                                              day_after.strftime('%d') + '_' + '*.dat'))   )
                
            # Add all files on the day to a single Dataframe
            inflow_slow = pd.DataFrame(columns=[ "RelativeHumidity",  "Pressure",  "Temperature", "RTD_Temp_11m", 'WS_15m'])
            for datafile in inflow_slow_files:
                inflow_slow = pd.concat( [inflow_slow if not inflow_slow.empty else None,   
                                          read_slow_data(datafile)] )     
            inflow_slow = inflow_slow.drop_duplicates().sort_index()
            inflow_slow = inflow_slow.rename(columns={
                "RelativeHumidity": "RH", "Pressure": "p", "Temperature": "Temp", "RTD_Temp_11m": "Temp_11m"})
            
            inflow_slow = inflow_slow[~inflow_slow.index.duplicated(keep='first')]  # remove lines where index occures more than once
            
            if (len(inflow)==0) & (len(inflow_slow)==0):
                print ('No inflow data')
                continue
            
          
            ### Read wake mast files 

            
            # mast1
            mast1_files = sorted(glob.glob(mast1_path +'Met_Mast_1_Sonics_' + year + '_' + month + '_' + day + '_' + '*.dat'))   #
            
            mast1 = pd.DataFrame()
            for datafile in mast1_files:
                mast1 = pd.concat( [mast1, read_sonic(datafile)] )      
            mast1 = mast1.drop_duplicates().sort_index()
            
            # mast2
            mast2_files = sorted(glob.glob(mast2_path +'Met_Mast_2_Sonics_' + year + '_' + month + '_' + day + '_' + '*.dat'))   #
            
            mast2 = pd.DataFrame()
            for datafile in mast2_files:
                mast2 = pd.concat( [mast2, read_sonic(datafile)] ) 
            mast2 = mast2.drop_duplicates().sort_index()
                        
            # mast3
            mast3_files = sorted(glob.glob(mast3_path +'Met_Mast_3_Sonics_' + year + '_' + month + '_' + day + '_' + '*.dat'))   #
            
            mast3 = pd.DataFrame()
            for datafile in mast3_files:
                mast3 = pd.concat( [mast3, read_sonic(datafile)] ) 
            mast3 = mast3.drop_duplicates().sort_index()    
            
            print ('Read ok')            
    
            
            #%% Modify data
            
            
            print ("Processing inflow mast ...")  
            
            if len(inflow) !=0:  
                
                ## filter outliers   
                filter_window = '60S'
                for channel in inflow.select_dtypes(include=[int, float]).columns:
                    inflow[channel] = inflow[channel].where( np.abs(inflow[channel] - inflow[channel].rolling(filter_window, center=True, min_periods=1).median() ) 
                                                            <= (5* inflow[channel].rolling(filter_window, center=True, min_periods=1).std() ) , np.nan)            
                
                ## wind speed and direction
                inflow = calc_wind(inflow)
       
        
        
                if debug == 1:
                    
                    # compare single wind components
                    plt.figure()                   
                    plt.figure(figsize=(17,10))
                    for height_col in [col for col in inflow.columns if 'U_ax_' in col]:    # loop over every mast height
                        plt.plot(inflow[height_col], label = height_col)
                    for height_col in [col for col in inflow.columns if 'V_ax_' in col]:    # loop over every mast height
                        plt.plot(inflow[height_col], label = height_col)
                    for height_col in [col for col in inflow.columns if 'W_ax_' in col]:    # loop over every mast height
                        plt.plot(inflow[height_col], label = height_col)
                    plt.grid(True)
                    plt.legend(loc=1)        


            if len(inflow_slow) !=0: 
                
                if np.diff(inflow_slow.index).mean() < pd.to_timedelta(filter_window):
                     inflow_slow.WS_15m = inflow_slow.WS_15m.where( np.abs(inflow_slow.WS_15m - inflow_slow.WS_15m.rolling(filter_window, center=True, min_periods=1).median() ) 
                                                             <= (5* inflow_slow.WS_15m.rolling(filter_window, center=True, min_periods=1).std() ) , np.nan)                                        
                
                inflow_slow.WS_15m = inflow_slow.WS_15m.where(inflow_slow.WS_15m < 30)
                inflow_slow.Temp = inflow_slow.Temp.where(inflow_slow.Temp < 100)
                filter_window = '60S'
                for channel in inflow_slow.select_dtypes(include=[int, float]).columns:
                    inflow_slow[channel] = inflow_slow[channel].where( np.abs(inflow_slow[channel] - inflow_slow[channel].rolling(filter_window, center=True, min_periods=1).median() ) 
                                                            <= (5* inflow_slow[channel].rolling(filter_window, center=True, min_periods=1).std() ) , np.nan)            


            ### Add low frequency data to sonic data
            inflow = pd.merge(inflow, inflow_slow, how='outer', left_index = True, right_index=True) 
            inflow.index.name = 'UTC'
            
            # Cut inflow data so that they are just one day
            inflow = inflow[pd.to_datetime(year+ month+ day + " 00:00:00") : 
                            pd.to_datetime(year+ month+ day + " 00:00:00") + pd.Timedelta(1,"d")]
            
                
            #     ### Temperature calibration and Ri calculation
            #     if {'U_ax_3m','U_ax_7m'}.issubset(inflow.columns):
            #         # Ri_b calculation with 7m and 3m temperature sensors
            #         if 'Temp_7m' in inflow:    #  (only for data after Nov 2022 when 7m Temperature sensor was installed)  if inflow.index[0] > pd.to_datetime('2022-11-17 00:00:00'): 
            #             inflow['Ri_b'] = Ri_bulk( inflow.Temp_7m, inflow.Temp_3m, 
            #                            inflow.U_ax_7m, inflow.U_ax_3m, inflow.V_ax_7m, inflow.V_ax_3m, 3.5 )     
            #         else:
            #             inflow['Ri_b'] = np.nan
                    
            #         # calibrate Sonic temperatures and use for Ri_b from Sonics
            #         def func(x, a,b):
            #             return a*x + b 
            #         # popt data based on Temperature calibration until 1/19/2023, see "02_Calibrate temperatures.py"
            #         popt7 = [  1.2829667 , -11.93461164]   
            #         inflow['Ts_7m_corr'] = func(inflow.Ts_7m, *popt7) # new Sonic-Tv          
            #         popt3 = [1.03067782, 0.58747216]
            #         inflow['Ts_3m_corr'] = func(inflow.Ts_3m, *popt3) # new Sonic-Tv
                    
            #         inflow['Ri_b_Sonic'] = Ri_bulk( inflow.Ts_7m_corr, inflow.Ts_3m_corr, 
            #                                        inflow.U_ax_7m, inflow.U_ax_3m, inflow.V_ax_7m, inflow.V_ax_3m, 3.5 )  
            #     else:
            #         inflow['Ri_b'] = np.nan
            #         inflow['Ts_7m_corr'] = np.nan
            #         inflow['Ts_3m_corr'] = np.nan
            #         inflow['Ri_b_Sonic'] = np.nan
                
                
            ### Calculate TKE in a defined rolling window for the inflow mast
            window_TKE = "10min"   
            
            for height_col in [col for col in inflow.columns if 'U_ax' in col]:    # loop over every mast height
            
                # define column names for TI and TKE
                U = height_col
                V = height_col.replace("U_ax", "V_ax")
                W = height_col.replace("U_ax", "W_ax")
                TKE = height_col.replace("U_ax", "TKE")
                TI_U = height_col.replace("U_ax", "TI")
                TI_w = height_col.replace("U_ax", "TI_w")
                # TI_uE = height_col.replace("U_ax", "TI_uE")
                # TI_vN = height_col.replace("U_ax", "TI_vN")
            
                # calculate TKE and TI
                inflow[TKE] = tke_time_window(inflow[U] ,inflow[V] ,inflow[W], time_window=window_TKE)
                inflow[TI_U] =  TI_time_window((inflow[U]**2 + inflow[V]**2)**0.5, (inflow[U]**2 + inflow[V]**2)**0.5, time_window=window_TKE)     
                inflow[TI_w] =  TI_time_window(inflow[W],(inflow[U]**2 + inflow[V]**2)**0.5, time_window=window_TKE)
                # inflow[TI_uE] =  TI_time_window(inflow[V],(inflow[U]**2 + inflow[V]**2)**0.5, time_window=window_TKE)
                # inflow[TI_vN] =  TI_time_window(inflow[U],(inflow[U]**2 + inflow[V]**2)**0.5, time_window=window_TKE)   
                
                
                
            ## Add flux data

            inflow['time'] = (inflow.index - inflow.index[0]) / np.timedelta64(1,'s')  

            fs = 20 # [Hz], sampling frequency

            window_fluxes = 60*20   # window for flux calc, in s


            # Initialize DataFrame for length scales (inflow)
            fluxes_inflow = pd.DataFrame()

                
            fluxes_inflow['H_S'] = np.nan 
            fluxes_inflow['Tau'] = np.nan 
            fluxes_inflow['R_f'] = np.nan  
            fluxes_inflow['zL'] = np.nan
            fluxes_inflow['L'] = np.nan


            # loop over 20 min segments
            for time, period in inflow.groupby( (window_fluxes/10.) * (inflow.time/(window_fluxes/10.)).round(-1)): # in window_fluxes

                maxlegs = int(window_fluxes*fs/2) # maximum lag for autocorrelation function     
                rho_Mast = rho(period.p, period.RH, period.Temp).mean()

                if len(period) > window_fluxes*fs/2: 
                    
                    # create new line with time stamp
                    fluxes_inflow = pd.concat([fluxes_inflow, pd.DataFrame({    
                         },
                        index = [ period.index[int(len(period)/2)]] )  ]) 
                    
                    # 7m height for inflow characteristics
                    fluxes_inflow['H_S'].iloc[-1] = HS(wE = period.W_ax_Top, theta = period.Ts_Top, Rho  =  rho_Mast) 
                    fluxes_inflow['Tau'].iloc[-1] = tau(period.W_ax_Top, period.wspd_Top, rho_Mast)  
                    fluxes_inflow['R_f'].iloc[-1] = Ri_flux( (period.Temp+period.Temp_11m)/2, period.Ts_Top, period.Ts_Low, 
                                                    period.U_ax_Top, period.U_ax_Low, period.V_ax_Top, period.V_ax_Low, 
                                                    period.W_ax_Top, 11-2.75 ) 
                    fluxes_inflow['zL'].iloc[-1] , fluxes_inflow['L'].iloc[-1]   = Obukhov_stability(Tref = period.Temp_11m, Tv= period.Ts_Top, 
                                                                      U = period.U_ax_Top, V = period.V_ax_Top, W = period.W_ax_Top, z=11)

            # remove repeated values
            for column in fluxes_inflow.drop(fluxes_inflow.columns,axis=1).columns: 
                #fluxes_inflow.loc[fluxes_inflow.duplicated([column]), column]=np.nan   # this drops all duplicates even if they occurr not consequtively
                index=fluxes_inflow[column].diff()== 0
                fluxes_inflow[column].loc[index]=np.nan
                
                
            inflow = inflow.merge(fluxes_inflow, left_index = True, right_index = True, how="outer")
                      



            # ### combine all wake masts
            masts = pd.merge(mast1.add_prefix('m1_'), mast2.add_prefix('m2_'), how='inner', left_index = True, right_index=True)    
            masts = pd.merge(masts, mast3.add_prefix('m3_'), how='inner', left_index = True, right_index=True)    
            masts.index.name = 'UTC'
                
            # if len(masts) !=0:
                
            #     ## Mast2 and 3 have specific faulty values
            #     for height_col in [col for col in masts.columns if 'U_ax_' in col]:    # loop over every mast height
            #         masts.loc[masts[height_col] == 3] = np.nan  
            #     for height_col in [col for col in masts.columns if 'V_ax_' in col]:    # loop over every mast height
            #         masts.loc[masts[height_col] == 33] = np.nan         
                    
                
            #     ## filter outliers (remove data that exceeds 5*std_dev in a 60s window)
            #     for channel in masts.columns:
            #         masts[channel] = masts[channel].where( np.abs(masts[channel] - masts[channel].rolling(filter_window, center=True, min_periods=1).median() ) 
            #                                               <= (5* masts[channel].rolling(filter_window, center=True, min_periods=1).std() ) , np.nan)   

               
            #     if debug == 1:                   
                    
            #         plt.figure(figsize=(17,10))
            #         for height_col in [col for col in masts.columns if 'U_ax_' in col]:    # loop over every mast height
            #             plt.plot(masts[height_col], label = height_col)
            #         for height_col in [col for col in masts.columns if 'V_ax_' in col]:    # loop over every mast height
            #             plt.plot(masts[height_col], label = height_col)
            #         for height_col in [col for col in masts.columns if 'W_ax_' in col]:    # loop over every mast height
            #             plt.plot(masts[height_col], label = height_col)
            #         plt.grid(True)
            #         plt.legend(loc=1)     
                  
                        
            #     ## wind speed and direction
            #     masts = calc_wind(masts)
                        

            #     # Calculate TKE in a defined rolling window for the wake masts
            #     for height_col in [col for col in masts.columns if 'U_ax_' in col]:    # loop over every mast height
                
            #         U = height_col
            #         V = height_col.replace("U_ax", "V_ax")
            #         W = height_col.replace("U_ax", "W_ax")
            #         TKE = height_col.replace("U_ax", "TKE")
            #         TI = height_col.replace("U_ax", "TI")
            #         TI_w = height_col.replace("U_ax", "TI_w")

            #         # calculate TKE and TI     
            #         masts[TKE] = tke_window(masts[U] ,masts[V] ,masts[W], fs=20, window=window_TKE)
            #         masts[TI] =  TI_window((masts[U]**2 + masts[V]**2)**0.5, (masts[U]**2 + masts[V]**2)**0.5, fs=20, window=window_TKE)  
            #         masts[TI_w] =  TI_window(masts[W], (masts[U]**2 + masts[V]**2)**0.5, fs=20, window=window_TKE)      


 
            


            #%% Resample , plot and save daily files
            
            res_freq = '1min'
            inflow_cut = inflow
            if len(inflow) !=0:  
                inflow_1min_cut = resample_sonic(inflow, res_freq)  


            # # Split DataFrame into daily chunks
            # daily_chunks = pd.Grouper(freq='D')
            # grouped_df = inflow.groupby(daily_chunks)
            
            # # Plot and Save each daily chunk separately
            # for group_name, inflow_cut in grouped_df:

            #     if len(masts) !=0: 
            #         masts_cut = masts[inflow_cut.index[0] : inflow_cut.index[-1]]
                
                
            #     ### Resample all Sonic data to 1 min data
            #     res_freq = '1min'
                
            #     if len(inflow_cut) !=0:  
            #         inflow_1min_cut = resample_sonic(inflow_cut, res_freq)  
            #     if len(masts_cut) !=0:  
            #         masts_1min_cut = resample_sonic(masts_cut, res_freq) 
                    
                    
            #     # add max and std values
            #     if len(inflow_cut) !=0:  
            #         inflow_max = inflow_cut.filter(regex='wspd').add_suffix('_max').resample(res_freq).max(numeric_only=True)
            #         inflow_std = inflow_cut.filter(regex='wspd').add_suffix('_std').resample(res_freq).std(numeric_only=True)   
            #         inflow_1min_cut = pd.concat([inflow_1min_cut, 
            #                                 inflow_max,
            #                                 inflow_std], axis=1)                
            #     if len(masts_cut) !=0:  
            #         masts_max = masts_cut.filter(regex='wspd').add_suffix('_max').resample(res_freq).max(numeric_only=True)
            #         masts_std = masts_cut.filter(regex='wspd').add_suffix('_std').resample(res_freq).std(numeric_only=True)   
            #         masts_1min_cut = pd.concat([masts_1min_cut, 
            #                                 masts_max,
            #                                 masts_std], axis=1)  
            #     else:
            #         masts_1min_cut = pd.DataFrame()

            #     # # Count number of nans for each wind component and in each 1 minute time step
            #     # inflow_nans = inflow_1min_cut.filter(regex='_ax_').isna().resample(res_freq).mean().add_suffix("_nans")
            #     # if len(masts) !=0:  
            #     #     masts_nans = masts_1min_cut.filter(regex='_ax_').isna().resample(res_freq).mean().add_suffix("_nans")
                
            
            
            
                ### Plot data            
                if plot == 1:
                    
                    
                    metar =     pd.read_csv('Tonopah_METAR.csv',
                                        index_col = 1,
                                        header = 0,    
                                        #skiprows = [0], 
                                        engine = 'c',
                                        on_bad_lines='warn', 
                                        parse_dates=True
                                            )   
                    
                    met = metar[inflow.index[0] :inflow.index[-1]  ]
                
                    # Timeseries
                    fig = plt.figure(figsize=(15,9))
                    fig.subplots_adjust(right=0.9)
                    plt.suptitle("{}".format(inflow_cut.index[0].date()))
                    ax1 = plt.subplot(3, 1, 2)
                    ax1.set_ylabel('Wind dir ($^\circ$)')
                    for height_col in [col for col in inflow_1min_cut.columns if ('wdir' in col)  & ('_m' not in col) & ('_s' not in col)]:    # loop over every inflow_1min_cut mast height   
                        ax1.plot(inflow_1min_cut[height_col],'.', label = height_col)
                    # for height_col in [col for col in masts_1min_cut.columns if ('wdir' in col)  & ('_m' not in col) & ('_s' not in col)]:    # loop over every wake mast height   
                    #     ax1.plot(masts_1min_cut[height_col],'.', label = height_col)
                    plt.plot(met.drct,"o", ms=3, color='black', label = "METAR") 
                    plt.legend(loc=1)
                    plt.grid()
                    
                    ax2 = plt.subplot(3, 1, 1, sharex=ax1)
                    plt.ylabel('Wind speed (m s$^{-1}$)')
                    plt.plot(inflow_1min_cut.WS_15m,'.', label = '15m', color="grey")                   
                    for height_col in [col for col in inflow_1min_cut.columns if ('wspd' in col)  & ('_m' not in col) & ('_s' not in col)]:    # loop over every mast height   
                        ax2.plot(inflow_1min_cut[height_col],'.', label = height_col)
                    # for height_col in [col for col in masts_1min_cut.columns if ('wspd' in col)  & ('_m' not in col) & ('_s' not in col)]:    # loop over every wake mast height   
                    #     ax2.plot(masts_1min_cut[height_col],'.', label = height_col)
                    plt.plot(met.sknt/1.94384,"o", ms=3, color='black', label = "METAR")  
                    plt.legend(loc=1)
                    plt.grid()
                    
                    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
                    plt.ylabel('Temp ($^\circ$)', color="C0")
                    plt.plot(inflow_1min_cut.Temp,".",label = 'Temp 2m', color="C0") 
                    try:
                        plt.plot(inflow_1min_cut.Temp_11m,'.',label = 'Temp 11m', color="lightblue")
                    except:
                        pass
                    plt.plot((met.tmpf-32)/1.8000,"o", ms=3, color='black', label = "") 
                    ax3.legend(loc=1).set_zorder(100)
                    ax4 = ax3.twinx()    
                    plt.plot(inflow_1min_cut.RH,'.',label = 'RH', color="C1") 
                    plt.plot(met.relh,"o", ms=5, color='black', label = "METAR") 
                    plt.ylabel('RH (%)', color="C1")  
                    ax5 = ax3.twinx()
                    ax5.spines.right.set_position(("axes", 1.05))
                    plt.plot(inflow_1min_cut.p,'.', label = 'p', color="C2")  
                    # plt.plot(met.mslp,"o", ms=3, color='black', label = "") # sea level pressure, not actual pressure
                    plt.ylabel('p (hPa)', color="C2")
                    plt.grid()    
                    
                    plt.tight_layout()
                
        
                    # # Histogram
                    # fig2 = plt.figure(figsize=(15,8))
                    # plt.suptitle("Histograms {} to {}".format(inflow_cut.index[0].date(), inflow_cut.index[-1].date() ))
                    # ax1 = plt.subplot(2, 1, 1)
                    # #ax1.set_title("Wind direction histogram {} to {}".format(inflow_cut.index[0].date(), inflow_cut.index[-1].date() ))
                    # for height_col in [col for col in inflow_cut.columns if 'wdir_' in col]:    # loop over every mast height   
                    #     plt.hist(inflow_cut[height_col].dropna(),label = height_col, alpha=0.5, bins=180, density=True) 
                    # for height_col in [col for col in masts_cut.columns if 'wdir_' in col]:    # loop over every mast height   
                    #     plt.hist(masts_cut[height_col].dropna(),label = height_col, alpha=0.5, bins=180, density=True) 
                    # plt.legend()
                    # plt.grid()
                    # plt.xlabel('Wind dir ($^\circ$)')
                        
                    # ax2 = plt.subplot(2, 1, 2)
                    # #ax2.set_title("Wind speed histogram {} to {}".format(inflow_cut.index[0].date(), inflow_cut.index[-1].date() ))
                    # for height_col in [col for col in inflow_cut.columns if 'wspd_' in col]:    # loop over every mast height   
                    #     plt.hist(inflow_cut[height_col].dropna(),label = height_col, alpha=0.5, bins=100, density=True) 
                    # for height_col in [col for col in masts_cut.columns if 'wspd_' in col]:    # loop over every mast height   
                    #     plt.hist(masts_cut[height_col].dropna(),label = height_col, alpha=0.5, bins=100, density=True) 
                    # plt.legend()
                    # plt.grid()
                    # plt.xlabel('Wind speed (m s$^{-1}$)')
                    # plt.tight_layout()
                    
                    
                    # # of nans
                    # fig3 = plt.figure(figsize=(15,8))
                    # plt.plot(inflow_nans.filter(regex='ax').mean(axis=1)
                    #          *100,
                    #          label = 'inflow')
                    # if len(masts_cut) !=0:
                    #     plt.plot(masts_nans.filter(regex='m1').mean(axis=1) 
                    #              *100, 
                    #              label = 'mast 1')
                    #     plt.plot(masts_nans.filter(regex='m2').mean(axis=1)
                    #              *100, 
                    #              label = 'mast 2')
                    #     plt.plot(masts_nans.filter(regex='m3').mean(axis=1)
                    #              *100, 
                    #              label = 'mast 3')
                    # plt.ylabel('% of NaNs in 10 min periods')
                    # plt.legend()                
                    # plt.tight_layout()
                        
    
    
                ### Save data files a pickle for faster reading
                if save == 1:
                    
                    print ("Saving data ...")  
    
                    if plot == 1:
                        fig.savefig(path_save+'Overview_{}_{:0>2}h_to_{}_{:0>2}h.png'.format(inflow_cut.index[0].date(),inflow_cut.index[0].hour , inflow_cut.index[-1].date(), inflow_cut.index[-1].hour), dpi=200) 
                        #fig2.savefig(path_save+'Histogram_{}_{}h_to_{}_{}h.png'.format(inflow_cut.index[0].date(),inflow_cut.index[0].hour , inflow_cut.index[-1].date(), inflow_cut.index[-1].hour), dpi=200) 
                        #fig3.savefig(path_save+'NaNs_{}_{}h_to_{}_{}h.png'.format(inflow_cut.index[0].date(),inflow_cut.index[0].hour , inflow_cut.index[-1].date(), inflow_cut.index[-1].hour), dpi=200)
                        plt.close('all')
                    
                    if len(inflow_cut) !=0:  
                        inflow_1min_cut.to_pickle(path_save+'Inflow_Mast_{}_{}_{:0>2}h_to_{}_{:0>2}h.pkl'.format(res_freq, inflow_cut.index[0].date(),inflow_cut.index[0].hour, inflow_cut.index[-1].date(),inflow_cut.index[-1].hour))
                        inflow_cut.to_pickle(path_save+'Inflow_Mast_20Hz_{}_{:0>2}h_to_{}_{:0>2}h.pkl'.format(inflow_cut.index[0].date(),inflow_cut.index[0].hour, inflow_cut.index[-1].date(),inflow_cut.index[-1].hour))
                        #inflow_nans.to_pickle(path_save+'Inflow_NaNs_{}_{}h_to_{}_{}h.pkl'.format(inflow_cut.index[0].date(),inflow_cut.index[0].hour, inflow_cut.index[-1].date(),inflow_cut.index[-1].hour))
                    # if len(masts_cut) !=0:  
                    #     masts_1min_cut.to_pickle(path_save+'Wake_Masts_{}_{}_{}h_to_{}_{}h.pkl'.format(res_freq, masts_cut.index[0].date(),masts_cut.index[0].hour, masts_cut.index[-1].date(),masts_cut.index[-1].hour))
                    #     masts_cut.to_pickle(path_save+'Wake_Masts_20Hz_{}_{}h_to_{}_{}h.pkl'.format(masts_cut.index[0].date(),masts_cut.index[0].hour, masts_cut.index[-1].date(),masts_cut.index[-1].hour))
                    #     #masts_nans.to_pickle(path_save+'Masts_NaNs_{}_{}h_to_{}_{}h.pkl'.format(inflow_cut.index[0].date(),inflow_cut.index[0].hour, inflow_cut.index[-1].date(),inflow_cut.index[-1].hour))

                    print ("ok ...")  
    
    
    
            # delete all data before processing next day
            del inflow,  inflow_cut, inflow_1min_cut,  # masts, masts_1min_cut, masts_cut     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


