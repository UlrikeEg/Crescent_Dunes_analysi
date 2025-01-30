import os
import pandas as pd
from nptdms import TdmsFile
import zipfile
import matplotlib.pyplot as plt
import glob
import sys

sys.path.append("C:/Users/uegerer/Desktop/NSO/NSO_data_processing")

from Functions_masts_CD import *
from Functions_loads_CD import *
from Functions_general import *






# Read Sonic data
# years   = [2024] 
# months  = [3] # 
# days    = [13,14,15]   # 

# path = 'Y:\Wind-data/Restricted/Projects/NSO/CrescentDunes_processed_met_tower_data_preliminary/'  # 'CD_processed_data_preliminary/'    #  
        

# inflow_file = path + 'Inflow_Mast_1min_2024-03-13_19h_to_2024-03-14_00h.pkl'

# ### Read data
# inflow = pd.read_pickle(inflow_file)
# inflow = inflow.sort_index()


path = 'Y:\Wind-data/Restricted/Projects/NSO/CrescentDunes-Loads/FastData/2024-03-19 to 2024-06-11/'  # 

loads = read_tdms_file_into_dataframe(path+'2024-05-21/CrescentDunes_2024_05_21_13_38_18_1Hz.tdms')


loads.index = pd.to_datetime(loads['LabVIEW Timestamp'], unit="s", origin=pd.Timestamp("1904-01-01"))




loads[[
       'H3_Support_Frame_Accel_1_X', 'H3_Support_Frame_Accel_1_Y',
       'H3_Support_Frame_Accel_1_Z', 'H3_Support_Frame_Accel_2_X',
       'H3_Support_Frame_Accel_2_Y', 'H3_Support_Frame_Accel_2_Z',
       'H3_Support_Frame_Accel_3_X', 'H3_Support_Frame_Accel_3_Y',
       'H3_Support_Frame_Accel_3_Z', 'H3_Support_Frame_Accel_4_X',
       'H3_Support_Frame_Accel_4_Y', 'H3_Support_Frame_Accel_4_Z',
       'H3_Mirror_Displacement_Top', 'H3_Mirror_Displacement_Bottom',
       'H3_Differential_Pressure_1', 'H3_Differential_Pressure_2',
       'H3_Differential_Pressure_3', 'H3_Azimuth ', 'H3_Elevation_Left ',
       'H3_Elevation_Right ', 'H3_Pedestal_Bend_1', 'H3_Pedestal_Bend_2 ',
       'H3_Pedestal_Torque', 'H3_Torque_Tube_Left', 'H3_Torque_Tube_Right',
       'H3_Pedestal_Axial ', 'H3_Support_Frame_Bending_Top',
       'H3_Support_Frame_Bending_Bottom']].plot()

loads.filter(regex="Accel").plot()



loads.filter(regex="Disp").plot()


loads.filter(regex="Diff").plot()

plt.figure()
plt.plot(loads.H3_Differential_Pressure_1, loads.H3_Differential_Pressure_2,".")
plt.plot(loads.H3_Differential_Pressure_1, loads.H3_Differential_Pressure_3,".")


loads[[ 'H3_Azimuth ', 'H3_Elevation_Left ','H3_Elevation_Right ']].plot()


# H1, H2, H3 = read_SCADA_file(paths = ['../SCADA/EMS1/', '../SCADA/EMS2/'], 
#                              filename_search_strings = ["Report*_MAY_2024*.csv"])
# plt.plot(H3.AngAzData, label = "SCADA Azimuth")
# plt.plot(H3.AngElData, label = "SCADA Elevation")
# plt.legend()


loads[[ 'H3_Pedestal_Bend_1', 'H3_Pedestal_Bend_2 ',
'H3_Pedestal_Torque', 'H3_Torque_Tube_Left', 'H3_Torque_Tube_Right',
'H3_Pedestal_Axial ', 'H3_Support_Frame_Bending_Top',
'H3_Support_Frame_Bending_Bottom']].plot()



# spectrum(loads, freq=20/60, channels=['H3_Support_Frame_Accel_4_X'], time_series=0)



# loads["H1_Wind_Speed"] = (loads['H1_Wind Speed U']**2 + loads['H1_Wind Speed V']**2)**0.5
# loads = loads[loads.H1_Wind_Speed<100]


# loads["H1_Wind_Speed"].plot()



# plt.figure()
# plt.plot(loads['H1_Wind_Speed'],  loads['H1_Support_Frame_Accel_1_X'],".", alpha = 0.5, label = "Heliostat 1, X-dir")
# plt.plot(loads['H1_Wind_Speed'],  loads['H1_Support_Frame_Accel_1_Y'],".", alpha = 0.5, label = "Heliostat 1, Y-dir")
# #plt.plot(loads['H1_Wind_Speed'],  loads['H1_Support_Frame_Accel_1_Z'],".", alpha = 0.5, label = "Heliostat 1, Z-dir")
# plt.plot(loads['H1_Wind_Speed'],  loads['H2_Support_Frame_Accel_1_X'],".", alpha = 0.5, label = "Heliostat 2, X-dir")
# plt.plot(loads['H1_Wind_Speed'],  loads['H2_Support_Frame_Accel_1_Y'],".", alpha = 0.5, label = "Heliostat 2, Y-dir")
# #plt.plot(loads['H1_Wind_Speed'],  loads['H2_Support_Frame_Accel_1_Z'],".", alpha = 0.5, label = "Heliostat 2, Z-dir")
# plt.legend()
# plt.grid()
# plt.ylabel("Acceleration")
# plt.xlabel('Inflow wind speed (m/s)')



# plt.rcParams.update({'font.size': 13})
# fig = plt.figure()
# ax = plt.subplot(2, 1, 1)
# plt.plot(loads['H1_Differential_Pressure_1'],".", ms = 1,alpha = 0.5, label = "Heliostat 1")
# plt.plot(-loads['H2_Differential_Pressure_1'],".", ms = 1,alpha = 0.5, label = "Heliostat 2")
# # plt.plot(loads['H2_Azimuth'],".", alpha = 0.5, label = "H1 Azimuth")
# # plt.plot(loads['H2_Elevation_Right '],".", alpha = 0.5, label = "H1 Elevation")
# # plt.plot(loads['H1_Azimuth'],".", alpha = 0.5, label = "H1 Azimuth")
# # plt.plot(loads['H1_Elevation_Right '],".", alpha = 0.5, label = "H1 Elevation")
# plt.legend(markerscale=10)
# plt.grid()
# fig.autofmt_xdate() 
# plt.ylabel ("Differential pressure (a.u.)")
# ax = plt.subplot(2, 1, 2, sharex = ax)
# #plt.plot(loads['H1_Wind_Speed'].resample("min").mean(),".", ms = 1,alpha = 0.5, label = "Heliostat 1", color="black")
# plt.plot(inflow.wspd_Mid, color="black")
# plt.grid()
# fig.autofmt_xdate() 
# plt.ylabel ("Wind speed at 5m (m/s)")
# plt.xlabel ("Day of March 2024 and time (UTC)")
# ax2 = ax.twinx()
# plt.plot(loads['H1_Elevation_Right '], color="red")
# plt.grid()
# plt.ylabel ("Elevation angle ($^\circ$)", color="red")
# plt.tight_layout()


# loads_filt = loads["2024-03-14 00:00:00":"2024-03-14 13:05:00"]

# plt.figure()
# for parameter in ['H1_Differential_Pressure_1']:
#     plt.plot(loads['H1_Wind_Speed'], loads[parameter],".", alpha = 0.3, label = "operational heliostat 1", ms = 1, zorder = 7)
#     plt.plot(loads_filt['H1_Wind_Speed'], loads_filt[parameter],".", alpha = 0.3, label = "stowed heliostat 1", ms = 0.5, zorder = 9)
# for parameter in ['H2_Differential_Pressure_1']:
#     plt.plot(loads['H1_Wind_Speed'], - loads[parameter],".", alpha = 0.3, label = "operational heliostat 2", ms = 1, zorder = 8)
#     plt.plot(loads_filt['H1_Wind_Speed'], - loads_filt[parameter],".", alpha = 0.3, label = "stowed heliostat 2", ms = 0.5, zorder = 10)
# plt.legend(markerscale=15)
# plt.grid()
# plt.ylabel('Differential pressure')
# plt.xlabel('Inflow wind speed (m/s)')



