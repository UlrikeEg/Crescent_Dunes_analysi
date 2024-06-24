import os
import pandas as pd
from nptdms import TdmsFile
import zipfile
import matplotlib.pyplot as plt

def read_tdms_file_into_dataframe(tdms_filename):
    try:
        tdms_file = TdmsFile(tdms_filename)
        data = {}
        for group in tdms_file.groups():
            for channel in group.channels():
                data[channel.name] = channel[:]
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        print(f"Error: {e}")

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

# if __name__ == "__main__":
#     concatenated_df = process_zip_in_same_directory()
#     print(concatenated_df)
    
    
# df = concatenated_df
# df.index = pd.to_datetime(df['LabVIEW Timestamp'], unit="s", origin=pd.Timestamp("1904-01-01"))


# df["H1_Wind_Speed"] = (df['H1_Wind Speed U']**2 + df['H1_Wind Speed V']**2)**0.5

# df = df[df.H1_Wind_Speed<100]

# df["H1_Wind_Speed"].plot()
# df["H1_Wind Direction "].plot()




# plt.figure()
# plt.plot(df['H1_Wind_Speed'],  df['H1_Support_Frame_Accel_1_X'],".", alpha = 0.5, label = "Heliostat 1, X-dir")
# plt.plot(df['H1_Wind_Speed'],  df['H1_Support_Frame_Accel_1_Y'],".", alpha = 0.5, label = "Heliostat 1, Y-dir")
# plt.plot(df['H1_Wind_Speed'],  df['H1_Support_Frame_Accel_1_Z'],".", alpha = 0.5, label = "Heliostat 1, Z-dir")
# plt.plot(df['H1_Wind_Speed'],  df['H2_Support_Frame_Accel_1_X'],".", alpha = 0.5, label = "Heliostat 2, X-dir")
# plt.plot(df['H1_Wind_Speed'],  df['H2_Support_Frame_Accel_1_Y'],".", alpha = 0.5, label = "Heliostat 2, Y-dir")
# plt.plot(df['H1_Wind_Speed'],  df['H2_Support_Frame_Accel_1_Z'],".", alpha = 0.5, label = "Heliostat 2, Z-dir")
# plt.legend()

# plt.grid()
# plt.ylabel("Acceleration")
# plt.xlabel('Inflow wind speed (m/s)')

# plt.plot(df['H1_Wind Speed W '],  df['H3_Support_Frame_Accel_1_Z'],".", alpha = 0.5)






# Read Sonic data
import glob
years   = [2024] 
months  = [3] # 
days    = [13,14,15]   # 

path = 'Y:\Wind-data/Restricted/Projects/NSO/CrescentDunes_processed_met_tower_data_preliminary/'  # 'CD_processed_data_preliminary/'    #  
        

inflow_file = path + 'Inflow_Mast_1min_2024-03-13_19h_to_2024-03-14_00h.pkl'

### Read data
inflow = pd.read_pickle(inflow_file)
inflow = inflow.sort_index()


path = 'Y:\Wind-data/Restricted/Projects/NSO/CrescentDunes-Loads/SlowData/'  # 'CD_processed_data_preliminary/'    #   

loads = read_tdms_file_into_dataframe(path+'CrescentDunes_2024_03_13_08_28_55_1Hz.tdms')
loads.index = pd.to_datetime(loads['LabVIEW Timestamp'], unit="s", origin=pd.Timestamp("1904-01-01"))

loads["H1_Wind_Speed"] = (loads['H1_Wind Speed U']**2 + loads['H1_Wind Speed V']**2)**0.5
loads = loads[loads.H1_Wind_Speed<100]


loads["H1_Wind_Speed"].plot()



plt.figure()
plt.plot(loads['H1_Wind_Speed'],  loads['H1_Support_Frame_Accel_1_X'],".", alpha = 0.5, label = "Heliostat 1, X-dir")
plt.plot(loads['H1_Wind_Speed'],  loads['H1_Support_Frame_Accel_1_Y'],".", alpha = 0.5, label = "Heliostat 1, Y-dir")
#plt.plot(loads['H1_Wind_Speed'],  loads['H1_Support_Frame_Accel_1_Z'],".", alpha = 0.5, label = "Heliostat 1, Z-dir")
plt.plot(loads['H1_Wind_Speed'],  loads['H2_Support_Frame_Accel_1_X'],".", alpha = 0.5, label = "Heliostat 2, X-dir")
plt.plot(loads['H1_Wind_Speed'],  loads['H2_Support_Frame_Accel_1_Y'],".", alpha = 0.5, label = "Heliostat 2, Y-dir")
#plt.plot(loads['H1_Wind_Speed'],  loads['H2_Support_Frame_Accel_1_Z'],".", alpha = 0.5, label = "Heliostat 2, Z-dir")
plt.legend()
plt.grid()
plt.ylabel("Acceleration")
plt.xlabel('Inflow wind speed (m/s)')



plt.rcParams.update({'font.size': 13})
fig = plt.figure()
ax = plt.subplot(2, 1, 1)
plt.plot(loads['H1_Differential_Pressure_1'],".", ms = 1,alpha = 0.5, label = "Heliostat 1")
plt.plot(-loads['H2_Differential_Pressure_1'],".", ms = 1,alpha = 0.5, label = "Heliostat 2")
# plt.plot(loads['H2_Azimuth'],".", alpha = 0.5, label = "H1 Azimuth")
# plt.plot(loads['H2_Elevation_Right '],".", alpha = 0.5, label = "H1 Elevation")
# plt.plot(loads['H1_Azimuth'],".", alpha = 0.5, label = "H1 Azimuth")
# plt.plot(loads['H1_Elevation_Right '],".", alpha = 0.5, label = "H1 Elevation")
plt.legend(markerscale=10)
plt.grid()
fig.autofmt_xdate() 
plt.ylabel ("Differential pressure (a.u.)")
ax = plt.subplot(2, 1, 2, sharex = ax)
#plt.plot(loads['H1_Wind_Speed'].resample("min").mean(),".", ms = 1,alpha = 0.5, label = "Heliostat 1", color="black")
plt.plot(inflow.wspd_Mid, color="black")
plt.grid()
fig.autofmt_xdate() 
plt.ylabel ("Wind speed at 5m (m/s)")
plt.xlabel ("Day of March 2024 and time (UTC)")
ax2 = ax.twinx()
plt.plot(loads['H1_Elevation_Right '], color="red")
plt.grid()
plt.ylabel ("Elevation angle ($^\circ$)", color="red")
plt.tight_layout()


loads_filt = loads["2024-03-14 00:00:00":"2024-03-14 13:05:00"]

plt.figure()
for parameter in ['H1_Differential_Pressure_1']:
    plt.plot(loads['H1_Wind_Speed'], loads[parameter],".", alpha = 0.3, label = "operational heliostat 1", ms = 1, zorder = 7)
    plt.plot(loads_filt['H1_Wind_Speed'], loads_filt[parameter],".", alpha = 0.3, label = "stowed heliostat 1", ms = 0.5, zorder = 9)
for parameter in ['H2_Differential_Pressure_1']:
    plt.plot(loads['H1_Wind_Speed'], - loads[parameter],".", alpha = 0.3, label = "operational heliostat 2", ms = 1, zorder = 8)
    plt.plot(loads_filt['H1_Wind_Speed'], - loads_filt[parameter],".", alpha = 0.3, label = "stowed heliostat 2", ms = 0.5, zorder = 10)
plt.legend(markerscale=15)
plt.grid()
plt.ylabel('Differential pressure')
plt.xlabel('Inflow wind speed (m/s)')



