import os
import pandas as pd
from nptdms import TdmsFile
import zipfile
import matplotlib.pyplot as plt
import sys
import glob
from itertools import product
import pickle



sys.path.append("../../NSO/NSO_data_processing")


from Functions_general import *
from Functions_loads_CD import *



#%% Read data


inflow = pd.read_pickle('Inflow_1min_2024-03-13_to_2025-01-06.pkl')
loads = pd.read_pickle("Loads_1min_2024-03-19_to_2024-06-12.pkl")


# t = read_tdms_file_into_dataframe('Y:\Wind-data/Restricted/Projects/NSO/CrescentDunes-Loads/SlowData/2024-05-17/CrescentDunes_2024_05_17_13_38_18_1Hz.tdms')



H1 = pd.read_pickle(f'SCADA_H1_1min_{H1.index[0].date()}_to_{H1.index[-1].date()}.pkl')
H2 = pd.read_pickle(f'SCADA_H2_1min_{H2.index[0].date()}_to_{H2.index[-1].date()}.pkl')
H3 = pd.read_pickle(f'SCADA_H3_1min_{H3.index[0].date()}_to_{H3.index[-1].date()}.pkl')



Index(['MS Excel Timestamp', 'LabVIEW Timestamp', 'Scan Errors', 'Late Scans',
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
       'H3_Support_Frame_Bending_Bottom', 'H2_Support_Frame_Accel_1_X',
       'H2_Support_Frame_Accel_1_Y', 'H2_Support_Frame_Accel_1_Z',
       'H2_Support_Frame_Accel_2_X', 'H2_Support_Frame_Accel_2_Y',
       'H2_Support_Frame_Accel_2_Z', 'H2_Support_Frame_Accel_3_X',
       'H2_Support_Frame_Accel_3_Y', 'H2_Support_Frame_Accel_3_Z',
       'H2_Support_Frame_Accel_4_X', 'H2_Support_Frame_Accel_4_Y',
       'H2_Support_Frame_Accel_4_Z', 'H2_Mirror_Displacement_Top',
       'H2_Mirror_Displacement_Bottom', 'H2_Differential_Pressure_1',
       'H2_Differential_Pressure_2', 'H2_Differential_Pressure_3',
       'H2_Azimuth', 'H2_Elevation_Left ', 'H2_Elevation_Right ',
       'H2_Pedestal_Bend_1', 'H2_Pedestal_Bend_2 ', 'H2_Pedestal_Torque',
       'H2_Torque_Tube_Left', 'H2_Torque_Tube_Right', 'H2_Pedestal_Axial ',
       'H2_Support_Frame_Bending_Top', 'H2_Support_Frame_Bending_Bottom',
       'H1_Support_Frame_Accel_1_X', 'H1_Support_Frame_Accel_1_Y',
       'H1_Support_Frame_Accel_1_Z', 'H1_Support_Frame_Accel_2_X',
       'H1_Support_Frame_Accel_2_Y', 'H1_Support_Frame_Accel_2_Z',
       'H1_Support_Frame_Accel_3_X', 'H1_Support_Frame_Accel_3_Y',
       'H1_Support_Frame_Accel_3_Z', 'H1_Support_Frame_Accel_4_X',
       'H1_Support_Frame_Accel_4_Y', 'H1_Support_Frame_Accel_4_Z',
       'H1_Mirror_Displacement_Top', 'H1_Mirror_Displacement_Bottom',
       'H1_Differential_Pressure_1', 'H1_Differential_Pressure_2',
       'H1_Differential_Pressure_3', 'H1_Azimuth', 'H1_Elevation_Left ',
       'H1_Elevation_Right ', 'H1_Wind Direction ', 'H1_Wind Speed W ',
       'H1_Wind Speed V', 'H1_Wind Speed U', 'H1_Pedestal_Bend_1',
       'H1_Pedestal_Bend_2 ', 'H1_Pedestal_Torque', 'H1_Torque_Tube_Left',
       'H1_Torque_Tube_Right', 'H1_Pedestal_Axial ',
       'H1_Support_Frame_Bending_Top', 'H1_Support_Frame_Bending_Bottom'],
      dtype='object')


loads[[col for col in loads.columns if "H3" in col]].plot()



    
