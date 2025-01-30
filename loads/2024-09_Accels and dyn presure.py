import os
import pandas as pd
from nptdms import TdmsFile
import zipfile
import matplotlib.pyplot as plt
import glob
import numpy as np



#%% Functions



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
    
    
def read_SCADA_file(path, filename):
    
    
    # filename = "Report_14_MAR_2024*.csv"
    
    # path = '../SCADA/EMS1/'  
    
    file = glob.glob(os.path.join(path, filename))[0]
    
    
    # Read datafile
    df = pd.read_csv(file, skiprows=[2], header=[0, 1], index_col=0, 
                     parse_dates=True, date_format ='%d-%b-%Y %H:%M:%S.%f PDT ' )
    
    # Convert the index from PDT to UTC
    df.index = df.index.tz_localize('America/Los_Angeles').tz_convert('UTC')
    
    
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


H1, H2, H3 = read_SCADA_file(path = '../SCADA/EMS1/', filename = "Report_14_MAR_2024*.csv")

    





#%% Read data

### Read Sonic data


path = 'Y:\Wind-data/Restricted/Projects/NSO/CrescentDunes_processed_met_tower_data_preliminary/'  # 'CD_processed_data_preliminary/'    #  
        

inflow_files = glob.glob(os.path.join(path, 'Inflow_Mast_1min_2024-03-*.pkl'))
                
inflow = read_and_combine_pickle_files(inflow_files)



### Read loads data




loads = pd.read_csv("loads_and_wind_1min_March.csv")
loads.index = pd.to_datetime(loads.iloc[:,0])

#%% Check data


# Check accelerations
accels = 0
if accels ==1 :
    
    plt.figure()
    for column in loads.filter(regex="H*_Support_Frame_Accel_").columns:
       plt.plot( loads[column],".", label = column)
        # plt.plot(loads['H1_Wind_Speed'], loads[column],".", alpha = 0.5, label = column)
    plt.legend()
    plt.grid()
    plt.ylabel("Acceleration")
    # plt.xlabel('Inflow wind speed (m/s)')
    
    
    plt.figure()
    for column in loads.filter(regex="H*_Differential_Pressure*").columns:
       plt.plot( loads[column],".", label = column)
        # plt.plot(loads['H1_Wind_Speed'], loads[column],".", alpha = 0.5, label = column)
    plt.legend()
    plt.grid()
    plt.ylabel("Differential pressure")


### Cut files

#loads = loads["2024-03-14 00:00:00":"2024-03-14 13:05:00"]

# Better to plot without index (remove later)
# loads = loads.reset_index(drop=True)
# loads = loads.dropna(how="all")
# loads = loads.reset_index(drop=True)

#%% Overview time series


                


# select good periods (steady wind, heliostat orientation)

plt.rcParams.update({'font.size': 12})
fig = plt.figure()

ax = plt.subplot(4, 1, 1)
#plt.plot(loads['H1_Wind_Speed'],".", ms = 1,alpha = 0.5, color="red", label = "loads")
plt.plot(loads.wspd_Mid,".", ms = 1, color="black", label = "Mid inflow mast")
fig.autofmt_xdate() 
plt.ylabel ("Wind speed at 5m (m/s)")
plt.xlabel ("Day of March 2024 and time (UTC)")
plt.grid()
plt.legend()

ax = plt.subplot(4, 1, 2, sharex = ax)
# plt.plot(loads['H2_Azimuth'],".", alpha = 0.5, label = "H2 Azimuth")
# plt.plot(loads['H2_Elevation_Right'],".", alpha = 0.5, label = "H2 Elevation")
plt.plot(loads.wdir_Mid, ".", ms = 1, color="black", label = "wind direction Mid inflow")   # inflow wind dir is verified against metar
# # plt.plot(loads["H1_Wind Direction "], ".", color="red", label = "wind direction loads")
plt.plot(loads['H1_Azimuth'],".", alpha = 0.5, label = "H1 Azimuth")
# plt.plot(loads['H2_Azimuth'],".", alpha = 0.5, label = "H2 Azimuth")

plt.plot(loads['H1_Elevation_Right'],".", alpha = 0.5, label = "H1 Elevation")
# plt.plot(loads['H2_Elevation_Right'],".", alpha = 0.5, label = "H2 Elevation")
# # plt.plot(loads['H1_Elevation_Left '],".", alpha = 0.5, label = "H1 Elevation left") # almost same as right


# Scada data
# plt.plot(H1.AngAzData,".", ms = 1, label = "SCADA H1 Azimuth")
# plt.plot(H1.AngElData-90, ".", ms = 1, label = "SCADA H1 Elev")
# plt.plot(H2.AngAzData,".", ms = 1, label = "SCADA H2 Azimuth")
# plt.plot(H2.AngElData,".", ms = 1, label = "SCADA H2 Elev")



plt.legend()
plt.grid()
plt.ylabel ("Angle ($^\circ$)")

ax = plt.subplot(4, 1, 3, sharex = ax)
for col in loads.filter(regex="H1_Differential").columns:
    plt.plot(loads[col],".", ms = 1,alpha = 0.5, label = col)
plt.ylabel ("Differential pressure (Pa)")
plt.legend(markerscale=10)
plt.grid()
fig.autofmt_xdate() 

ax = plt.subplot(4, 1, 4, sharex = ax)
for col in loads.filter(regex="H1_Support_Frame_Accel").columns:
    plt.plot(loads[col],".", ms = 1,alpha = 0.5, label = col)
plt.ylabel ("Acceleration (g)")
plt.legend(markerscale=10)
plt.grid()
fig.autofmt_xdate() 

plt.tight_layout()







#%% Make plots






loads_operating = loads.where(loads.H1_Elevation_Right.round() != 0)
loads_stow = loads.where(loads.H1_Elevation_Right.round() == 0)

plt.figure()
for parameter in ['H1_Differential_Pressure_1']:
    plt.plot(loads_operating['H1_Wind_Speed'], loads_operating[parameter],".", alpha = 0.9, label = "operational heliostat 1", ms = 1, zorder = 7)
    plt.plot(loads_stow['H1_Wind_Speed'], loads_stow[parameter],".",  alpha = 0.9, label = "stowed heliostat 1", ms = 1, zorder = 9)
for parameter in ['H2_Differential_Pressure_1']:
    plt.plot(loads_operating['H1_Wind_Speed'], - loads_operating[parameter],".",  alpha = 0.9, label = "operational heliostat 2", ms = 1, zorder = 8)
    plt.plot(loads_stow['H1_Wind_Speed'], - loads_stow[parameter],".",  alpha = 0.9, label = "stowed heliostat 2", ms = 1, zorder = 10)
plt.legend(markerscale=5)
plt.grid()
plt.ylabel('Differential pressure (Pa)')
plt.xlabel('Inflow wind speed (m/s)')



plt.figure()
for parameter in loads.filter(regex="H1_Differential_Pressure*").columns:
    plt.plot(loads['H1_Elevation_Right'], loads[parameter],".", alpha = 0.9, label = parameter, ms = 1, zorder = 7)
for parameter in loads.filter(regex="H2_Differential_Pressure*").columns:
    plt.plot(loads['H2_Elevation_Right'], - loads[parameter],".",  alpha = 0.9, label = parameter, ms = 1, zorder = 8)

plt.legend(markerscale=5)
plt.grid()
plt.ylabel('Differential pressure')
plt.xlabel('Elevation (deg)')



