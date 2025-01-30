import os
import pandas as pd
from nptdms import TdmsFile
import zipfile
import matplotlib.pyplot as plt
import glob
import numpy as np
import sys



sys.path.append("../../NSO/NSO_data_processing")

from Functions_general import *



#%% 


        



H1, H2, H3 = read_SCADA_file(path = '../SCADA/EMS1/', filename = "Report_14_MAR_2024*.csv")

compare_EMS1_EMS2=0
if compare_EMS1_EMS2==1:
    
    plt.figure()
    
    path = '../SCADA/EMS1/'
    ems1_files = glob.glob(os.path.join(path, 'Report_*.csv'))
    
    for file in ems1_files:
        H1, H2, H3 = read_SCADA_file(path = './', filename = file)
        for col in [ 'AngAzData', 'AngElData']:
            plt.plot(H1[col], label = "", color = "red")
            
    path = '../SCADA/EMS2/'
    ems1_files = glob.glob(os.path.join(path, 'Report_*.csv'))
    
    for file in ems1_files:
        H1, H2, H3 = read_SCADA_file(path = './', filename = file)
        for col in [ 'AngAzData', 'AngElData']:
            plt.plot(H1[col], label = "", color = "black")
            
    plt.plot(np.nan, np.nan, label = "EMS1", color = "red" )
    plt.plot(np.nan, np.nan, label = "EMS2", color = "black" )
    plt.legend()  
    plt.ylabel ("Angle ($^\circ$)")
        
"""
'AzCmd: fixed value
'ElCmd': fixed value
'AzData': actual data?
'ElData': actual data?
'State':
'XAim_ENU':
'YAim_ENU':
'ZAim_ENU':
'AngAzCmd':
'AngElCmd':
'AngAzData': 
'AngElData': 
"""



plot_scada = 0

if plot_scada == 1:
    
    plt.figure()
    for col in [ 'AngAzData', 'AngElData']:
        plt.plot(H1[col], label = col)
    plt.legend() 
    plt.ylabel ("Angle ($^\circ$)")
    
    # Identify the start and end times for each state, including interruptions
    state_changes = H1[['State']].reset_index()
    state_changes['Next Time'] = state_changes['index'].shift(-1)
    state_changes = state_changes.dropna().reset_index(drop=True)
    
    # Add background shading for each interval
    for i, row in state_changes.iterrows():
        plt.axvspan(row['index'], row['Next Time'], color='gray', alpha=0.2, label=row['State'])
    
    # Handle overlapping labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    
    
    plt.figure()
    plt.suptitle("Each color is a helisotat in the EMS1 file")
    vars_to_plot = [#'AzCmd', 'ElCmd', 
                   # 'AzData', 'ElData', 
                    'State', 'XAim_ENU',
           'YAim_ENU', 'ZAim_ENU', 'AngAzCmd', 'AngElCmd', 'AngAzData',
           'AngElData']
    n_subplots = len(vars_to_plot)+1
    nrows = int(n_subplots**0.5)
    ncols = nrows if nrows * nrows >= n_subplots else nrows + 1
    for i,variable in enumerate(vars_to_plot, start=1):
        plt.subplot(nrows, ncols, i)
        plt.title(variable)
        for H, align in zip([H1, H2, H3], ['left', 'mid', 'right']):
            plt.hist(H[variable], alpha = 0.5 , align = align , bins=50) #)
    plt.tight_layout()
    
    





#%% Read data

## Read Sonic data


# path = 'Y:\Wind-data/Restricted/Projects/NSO/CrescentDunes_processed_met_tower_data_preliminary/'  # 'CD_processed_data_preliminary/'    #  
        

# inflow_files = glob.glob(os.path.join(path, 'Inflow_Mast_20Hz_2024-03-*.pkl'))
                
# inflow = read_and_combine_pickle_files(inflow_files)
# inflow = inflow.resample("S").first()



# ### Read loads data

# path = 'Y:\Wind-data/Restricted/Projects/NSO/CrescentDunes-Loads/SlowData/'  
# fast_path = 'Y:\Wind-data/Restricted/Projects/NSO/CrescentDunes-Loads/FastData/'

# ## single file
# #loads = read_tdms_file_into_dataframe(path+'CrescentDunes_2024_03_13_08_28_55_1Hz.tdms')
# #loads = read_tdms_file_into_dataframe(path+'2024-03-19 to 2024-06-11/2024-03-19/CrescentDunes_2024_03_19_16_17_29_1Hz.tdms')
# #loads = read_tdms_file_into_dataframe(path+'CrescentDunes_2024_03_14_08_28_55_1Hz.tdms')
# # from May on only H3, in March H1 + H2 available
# #loads = read_tdms_file_into_dataframe(path+'2024-03-19 to 2024-06-11/2024-06-11/CrescentDunes_2024_06_11_13_38_18_1Hz.tdms')

# # Multiple files
# # file_list = [path+'CrescentDunes_2024_03_13_08_28_55_1Hz.tdms',
# #              path+'2024-03-19 to 2024-06-11/2024-03-19/CrescentDunes_2024_03_19_16_17_29_1Hz.tdms']
# file_list = glob.glob(os.path.join(path, 'CrescentDunes_2024_03_13*.tdms')) + \
#             glob.glob(os.path.join(path, '2024-03-19 to 2024-06-11/2024-03-*/CrescentDunes*.tdms'))  # 14th files is corrupted
            
# # fast files in March
# file_list = glob.glob(os.path.join(fast_path, '2024-03-13/CrescentDunes*.tdms')) + \
#             glob.glob(os.path.join(fast_path, '2024-03-14/CrescentDunes*.tdms')) + \
#             glob.glob(os.path.join(fast_path, '2024-03-19 to 2024-06-11/2024-03-*/CrescentDunes*.tdms')) 
                        
            
            
# # file_list = glob.glob(os.path.join(path, '2024-03-19 to 2024-06-11/2024-05-*/CrescentDunes*.tdms'))  # 14th files is corrupted
            
# # file_list = glob.glob(os.path.join(fast_path, '2024-03-14/CrescentDunes_2024_03_14*.tdms'))


# loads = read_multiple_tdms_file_into_dataframe(file_list)

# # Add wind speed column
# loads["H1_Wind_Speed"] = (loads['H1_Wind Speed U']**2 + loads['H1_Wind Speed V']**2)**0.5
# loads = loads[loads.H1_Wind_Speed<100]


# # Clean up the column names by stripping leading and trailing spaces
# loads.columns = loads.columns.str.strip()

# # drop columns
# loads = loads.drop(columns=['MS Excel Timestamp','LabVIEW Timestamp','Scan Errors', 'Late Scans'])




# """
# Index(['MS Excel Timestamp', 'LabVIEW Timestamp', 'Scan Errors', 'Late Scans',
#         'H3_Support_Frame_Accel_1_X', 'H3_Support_Frame_Accel_1_Y',
#         'H3_Support_Frame_Accel_1_Z', 'H3_Support_Frame_Accel_2_X',
#         'H3_Support_Frame_Accel_2_Y', 'H3_Support_Frame_Accel_2_Z',
#         'H3_Support_Frame_Accel_3_X', 'H3_Support_Frame_Accel_3_Y',
#         'H3_Support_Frame_Accel_3_Z', 'H3_Support_Frame_Accel_4_X',
#         'H3_Support_Frame_Accel_4_Y', 'H3_Support_Frame_Accel_4_Z',
#         'H3_Mirror_Displacement_Top', 'H3_Mirror_Displacement_Bottom',
#         'H3_Differential_Pressure_1', 'H3_Differential_Pressure_2',
#         'H3_Differential_Pressure_3', 'H3_Azimuth ', 'H3_Elevation_Left ',
#         'H3_Elevation_Right ', 'H3_Pedestal_Bend_1', 'H3_Pedestal_Bend_2 ',
#         'H3_Pedestal_Torque', 'H3_Torque_Tube_Left', 'H3_Torque_Tube_Right',
#         'H3_Pedestal_Axial ', 'H3_Support_Frame_Bending_Top',
#         'H3_Support_Frame_Bending_Bottom', 'H2_Support_Frame_Accel_1_X',
#         'H2_Support_Frame_Accel_1_Y', 'H2_Support_Frame_Accel_1_Z',
#         'H2_Support_Frame_Accel_2_X', 'H2_Support_Frame_Accel_2_Y',
#         'H2_Support_Frame_Accel_2_Z', 'H2_Support_Frame_Accel_3_X',
#         'H2_Support_Frame_Accel_3_Y', 'H2_Support_Frame_Accel_3_Z',
#         'H2_Support_Frame_Accel_4_X', 'H2_Support_Frame_Accel_4_Y',
#         'H2_Support_Frame_Accel_4_Z', 'H2_Mirror_Displacement_Top',
#         'H2_Mirror_Displacement_Bottom', 'H2_Differential_Pressure_1',
#         'H2_Differential_Pressure_2', 'H2_Differential_Pressure_3',
#         'H2_Azimuth', 'H2_Elevation_Left ', 'H2_Elevation_Right ',
#         'H2_Pedestal_Bend_1', 'H2_Pedestal_Bend_2 ', 'H2_Pedestal_Torque',
#         'H2_Torque_Tube_Left', 'H2_Torque_Tube_Right', 'H2_Pedestal_Axial ',
#         'H2_Support_Frame_Bending_Top', 'H2_Support_Frame_Bending_Bottom',
#         'H1_Support_Frame_Accel_1_X', 'H1_Support_Frame_Accel_1_Y',
#         'H1_Support_Frame_Accel_1_Z', 'H1_Support_Frame_Accel_2_X',
#         'H1_Support_Frame_Accel_2_Y', 'H1_Support_Frame_Accel_2_Z',
#         'H1_Support_Frame_Accel_3_X', 'H1_Support_Frame_Accel_3_Y',
#         'H1_Support_Frame_Accel_3_Z', 'H1_Support_Frame_Accel_4_X',
#         'H1_Support_Frame_Accel_4_Y', 'H1_Support_Frame_Accel_4_Z',
#         'H1_Mirror_Displacement_Top', 'H1_Mirror_Displacement_Bottom',
#         'H1_Differential_Pressure_1', 'H1_Differential_Pressure_2',
#         'H1_Differential_Pressure_3', 'H1_Azimuth', 'H1_Elevation_Left ',
#         'H1_Elevation_Right ', 'H1_Wind Direction ', 'H1_Wind Speed W ',
#         'H1_Wind Speed V', 'H1_Wind Speed U', 'H1_Pedestal_Bend_1',
#         'H1_Pedestal_Bend_2 ', 'H1_Pedestal_Torque', 'H1_Torque_Tube_Left',
#         'H1_Torque_Tube_Right', 'H1_Pedestal_Axial ',
#         'H1_Support_Frame_Bending_Top', 'H1_Support_Frame_Bending_Bottom'],
#       dtype='object')

# """






# # resample (loads slow 1s data is samples at full seconds)
# loads = loads.resample("S").first()

# loads = loads.dropna(how="all")


# #inflow = inflow[:"2024-03-20 10:00:00"]
# loads = pd.merge(loads, inflow, left_index=True, right_index=True, how="inner")


# loads.to_csv("loads_and_wind_1s_March.csv")
 
loads = pd.read_csv("loads_and_wind_1s_March.csv", index_col=0, parse_dates=True)

#%% Check data




# Compare wind speeds from inflow and loads
compare_wind_speeds =  0 
if compare_wind_speeds ==1 :
    
    
    plt.figure()
    plt.plot(loads["H1_Wind_Speed"], color="black", label = "loads")  # .resample("min").mean()
    plt.plot(inflow.wspd_Low, label = "inflow low")
    plt.grid()
    plt.legend()
    
    
### Cut files

#loads = loads["2024-03-14 00:00:00":"2024-03-14 13:05:00"]

# Better to plot without index (remove later)
# loads = loads.reset_index(drop=True)
# loads = loads.dropna(how="all")
# loads = loads.reset_index(drop=True)

#%% Calculate load coefficients



H1.index = H1.index.tz_localize(None).round("S")
loads = pd.merge(loads, H1[['State', 'AngAzData', 'AngElData']], left_index=True, right_index=True, suffixes=('', '_H1'))
                

# Calculate side angle
loads["side_angle"] = (loads["wdir_Mid"] - loads["AngAzData"] ) % 360 - 180


# Correct axial strain gage
loads_corr = loads["2024-03-14 10:00:00":"2024-03-14 13:10:00"]  # period at stow with relatively low winds

loads = loads["2024-03-14 00:00:00":"2024-03-15 00:00:00"]

axial_slope = 8.2075E+04  # kN/V/V, info from Scott
axial_offset = loads_corr.H1_Pedestal_Axial.median()	# kN/V/V, averaged offset in calm stow period


loads["H1_F_Lift"] = (loads.H1_Pedestal_Axial- axial_offset) * axial_slope 

## Add load coefficients
wind_limit_coeff = 2
loads = add_load_coefficients_CD(loads, wind_speed_limit=3)

# Exclude load coefficients for fast angle changes

loads["delta_Az"] = loads.AngAzData.diff().fillna(0, limit=1)   / loads.index.diff().fillna(pd.to_timedelta(0, "S")).total_seconds().astype(int)
loads["delta_El"] = loads.AngElData.diff().fillna(0, limit=1)   / loads.index.diff().fillna(pd.to_timedelta(0, "S")).total_seconds().astype(int)

plt.figure()
plt.hist(loads.delta_Az.dropna(), bins=100, range=(-0.1, 0.1), alpha=0.5)
plt.hist(loads.delta_El.dropna(), bins=100, range=(-0.1, 0.1), alpha=0.5)
plt.hist(loads.where(((loads['delta_Az'] > -0.005) & (loads['delta_Az'] < 0.005))).delta_Az.dropna(), bins=100, range=(-0.1, 0.1), alpha=0.5)
plt.semilogy()


# for column in loads[[col for col in loads.columns if ( ('C_' in col) or ('Cfx' in col))]]:




# plt.figure()
# plt.plot(loads.H1_C_lift)

# loads.loc[abs(loads['delta_Az']) >= 0.005, 'H1_C_lift'] = None
# loads.loc[abs(loads['delta_El']) >= 0.005, 'H1_C_lift'] = None

# loads['H1_C_lift'] = loads['H1_C_lift'].where(abs(loads['delta_Az']) < 0.005)
# loads['H1_C_lift'] = loads['H1_C_lift'].where(abs(loads['delta_El']) < 0.005)


loads = loads[~((loads.index >= "2024-03-14 13:07:00") & (loads.index <= "2024-03-14 14:02:00"))]




             

# Calculate side angle
loads["side_angle"] = (loads["wdir_Mid"] - loads["AngAzData"] ) % 360 - 180

plot_side_angle=0
if plot_side_angle==1:
    
    
    plt.figure()
    plt.plot(test,".")
    plt.plot(loads["wdir_Mid"],".")
    plt.plot( loads.AngAzData,".")
    plt.plot( loads.AngElData,".")
    
    
# set flag when inflow wind is blocked by other heliostats
loads["inflow_free"] = np.where((loads['wdir_Mid'] > 246) | (loads['wdir_Mid'] < 53), 1, 0)

loads = loads.where(loads.inflow_free==1)    


    

    
    

# select good periods (steady wind, heliostat orientation)

plt.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(16,9))

ax = plt.subplot(5, 1, 1)
#plt.plot(loads['H1_Wind_Speed'],".", ms = 1,alpha = 0.5, color="red", label = "loads")
plt.plot(loads.wspd_Mid,".", ms = 5, color="black", label = "Mid inflow mast")
fig.autofmt_xdate() 
plt.ylabel ("Wind speed at 5m (m/s)")
plt.xlabel ("Day of March 2024 and time (UTC)")
plt.grid()
plt.legend()

ax = plt.subplot(5, 1, 2, sharex = ax)
# plt.plot(loads['H2_Azimuth'],".", alpha = 0.5, label = "H2 Azimuth")
# plt.plot(loads['H2_Elevation_Right'],".", alpha = 0.5, label = "H2 Elevation")
plt.plot(loads.wdir_Mid, ".", ms = 5, color="black", label = "wind direction Mid inflow")   # inflow wind dir is verified against metar
# # plt.plot(loads["H1_Wind Direction "], ".", color="red", label = "wind direction loads")
# plt.plot(loads['H1_Azimuth'],".", alpha = 0.5, label = "H1 Azimuth")
# plt.plot(loads['H2_Azimuth'],".", alpha = 0.5, label = "H2 Azimuth")

plt.plot(loads['H1_Elevation_Right'] + 180,".", alpha = 0.5, label = "H1 Elevation + 180$^\circ$")
# plt.plot(loads['H2_Elevation_Right'],".", alpha = 0.5, label = "H2 Elevation")
# # plt.plot(loads['H1_Elevation_Left '],".", alpha = 0.5, label = "H1 Elevation left") # almost same as right


# Scada data
plt.plot(loads.AngAzData,".", ms = 5, label = "SCADA H1 Azimuth")
# plt.plot(loads.AngElData,".", ms = 5, label = "SCADA H1 Elevation")
# plt.plot(H1.AngAzData,".", ms = 1, label = "SCADA H1 Azimuth")
# plt.plot(H1.AngElData-90, ".", ms = 1, label = "SCADA H1 Elev")
# plt.plot(H2.AngAzData,".", ms = 1, label = "SCADA H2 Azimuth")
# plt.plot(H2.AngElData,".", ms = 1, label = "SCADA H2 Elev")



plt.legend()
plt.grid()
plt.ylabel ("Angle ($^\circ$)")

ax = plt.subplot(5, 1, 3, sharex = ax)
# plt.plot(loads['H1_Differential_Pressure_1'],".", ms = 1,alpha = 0.5, label = "Heliostat 1")
# plt.plot(-loads['H2_Differential_Pressure_1'],".", ms = 1,alpha = 0.5, label = "Heliostat 2")
# plt.plot(loads['H1_Differential_Pressure_1'],".", ms = 1,alpha = 0.5, label = "Heliostat 1")
# plt.plot(-loads['H2_Differential_Pressure_1'],".", ms = 1,alpha = 0.5, label = "Heliostat 2")
# plt.ylabel ("Differential pressure (Pa)")
for strain in [
#         'H1_Pedestal_Bend_1',
# 'H1_Pedestal_Bend_2', 'H1_Pedestal_Torque', 'H1_Torque_Tube_Left',
# 'H1_Torque_Tube_Right',
'H1_Pedestal_Axial',
# 'H1_Support_Frame_Bending_Top', 'H1_Support_Frame_Bending_Bottom'
]:
    plt.plot(loads[strain],".", ms = 2,alpha = 1, label = strain)
plt.ylabel ("raw strain (V/V)")
plt.legend(markerscale=3)
plt.grid()
fig.autofmt_xdate() 

ax = plt.subplot(5, 1, 4, sharex = ax)
for col in ['H1_F_Lift']:  #, 'H2_C_lift', 'H3_C_lift'
    plt.plot(loads[col],".", ms = 2,alpha = 1, label = col)
plt.ylabel ("Lift force (kN)")
plt.legend(markerscale=3)
plt.grid()
fig.autofmt_xdate() 

ax = plt.subplot(5, 1, 5, sharex = ax)
for col in ['H1_C_lift']:  #, 'H2_C_lift', 'H3_C_lift'
    plt.plot(loads[col],".", ms = 2,alpha = 1, label = col)
plt.ylabel ("Load coefficient")
plt.legend(markerscale=3)
plt.grid()
fig.autofmt_xdate() 


plt.tight_layout()

loads_res = loads.select_dtypes(include=['number']).resample("60s").median()
loads_res.wdir_Mid = np.degrees(np.arctan2( - loads_res["V_ax_Mid"], loads_res["U_ax_Mid"])) + 180




plt.figure()
for col in ['H1_C_lift']: # , 'H2_C_lift', 'H3_C_lift']:
    cb = plt.scatter(loads_res['H1_Elevation_Right'], loads_res[col], s=4,c=loads_res.side_angle,label = col, vmin=-100, 
        vmax=100, cmap="coolwarm"  )
plt.ylabel ("Load coefficient")
plt.xlabel ("Elevation")
plt.legend(markerscale=3)
plt.grid()
plt.colorbar(cb, label='Side Angle')  
fig.autofmt_xdate() 



# Matt_data = loads[["wspd_Mid", "wdir_Mid", "AngAzData", "H1_Elevation_Right", "H1_F_Lift", "H1_C_lift", "side_angle"]]

# Matt_data.to_csv('Lift_coefficient_data_CD_for_Matt_9-10-24.csv', index=True) 





#%% Single pltos

plt.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(15,5))

ax = plt.subplot(1, 1, 1)
#plt.plot(loads['H1_Wind_Speed'],".", ms = 1,alpha = 0.5, color="red", label = "loads")
# plt.plot(loads.wspd_Mid,".", ms = 5, color="black", label = "Mid inflow mast")
# fig.autofmt_xdate() 
# plt.ylabel ("Wind speed at 5m (m/s)")
plt.xlabel ("Day of March 2024 and time (UTC)")
# plt.grid()
# plt.legend()

# ax = plt.subplot(5, 1, 2, sharex = ax)
# plt.plot(loads['H2_Azimuth'],".", alpha = 0.5, label = "H2 Azimuth")
# plt.plot(loads['H2_Elevation_Right'],".", alpha = 0.5, label = "H2 Elevation")
plt.plot(loads.wdir_Mid, ".", ms = 5, color="black", label = "wind direction Mid inflow")   # inflow wind dir is verified against metar
# # plt.plot(loads["H1_Wind Direction "], ".", color="red", label = "wind direction loads")
# plt.plot(loads['H1_Azimuth'],".", alpha = 0.5, label = "H1 Azimuth")
# plt.plot(loads['H2_Azimuth'],".", alpha = 0.5, label = "H2 Azimuth")

plt.plot(loads['H1_Elevation_Right'] + 180,".", alpha = 0.5, label = "H1 Elevation")  #+ 180$^\circ$
# plt.plot(loads['H2_Elevation_Right'],".", alpha = 0.5, label = "H2 Elevation")
# # plt.plot(loads['H1_Elevation_Left '],".", alpha = 0.5, label = "H1 Elevation left") # almost same as right


# Scada data
plt.plot(loads.AngAzData,".", ms = 5, label = "SCADA H1 Azimuth")
# plt.plot(loads.AngElData,".", ms = 5, label = "SCADA H1 Elevation")
# plt.plot(H1.AngAzData,".", ms = 1, label = "SCADA H1 Azimuth")
# plt.plot(H1.AngElData-90, ".", ms = 1, label = "SCADA H1 Elev")
# plt.plot(H2.AngAzData,".", ms = 1, label = "SCADA H2 Azimuth")
# plt.plot(H2.AngElData,".", ms = 1, label = "SCADA H2 Elev")



plt.legend(markerscale=5)
plt.grid()
plt.ylabel ("Angle ($^\circ$)")

# ax = plt.subplot(5, 1, 3, sharex = ax)
# # plt.plot(loads['H1_Differential_Pressure_1'],".", ms = 1,alpha = 0.5, label = "Heliostat 1")
# # plt.plot(-loads['H2_Differential_Pressure_1'],".", ms = 1,alpha = 0.5, label = "Heliostat 2")
# # plt.plot(loads['H1_Differential_Pressure_1'],".", ms = 1,alpha = 0.5, label = "Heliostat 1")
# # plt.plot(-loads['H2_Differential_Pressure_1'],".", ms = 1,alpha = 0.5, label = "Heliostat 2")
# # plt.ylabel ("Differential pressure (Pa)")
# for strain in [
# #         'H1_Pedestal_Bend_1',
# # 'H1_Pedestal_Bend_2', 'H1_Pedestal_Torque', 'H1_Torque_Tube_Left',
# # 'H1_Torque_Tube_Right',
# 'H1_Pedestal_Axial',
# # 'H1_Support_Frame_Bending_Top', 'H1_Support_Frame_Bending_Bottom'
# ]:
#     plt.plot(loads[strain],".", ms = 2,alpha = 1, label = strain)
# plt.ylabel ("raw strain (V/V)")
# plt.legend(markerscale=3)
# plt.grid()
# fig.autofmt_xdate() 

# ax = plt.subplot(5, 1, 4, sharex = ax)
# for col in ['H1_F_Lift']:  #, 'H2_C_lift', 'H3_C_lift'
#     plt.plot(loads[col],".", ms = 2,alpha = 1, label = col)
# plt.ylabel ("Lift force (kN)")
# plt.legend(markerscale=3)
# plt.grid()
# fig.autofmt_xdate() 

# ax = plt.subplot(5, 1, 5, sharex = ax)
# for col in ['H1_C_lift']:  #, 'H2_C_lift', 'H3_C_lift'
#     plt.plot(loads[col],".", ms = 2,alpha = 1, label = col)
# plt.ylabel ("Load coefficient")
# plt.legend(markerscale=3)
# plt.grid()


plt.xlim(loads.index[0], loads.index[-1])

fig.autofmt_xdate() 


plt.tight_layout()

fig.savefig('C:/Users/uegerer/Desktop/Load_coeff_time_series2.pdf', dpi=300)




