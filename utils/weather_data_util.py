import os
import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt

def load_bj_grid_meo_data(useful_stations):
    '''
    csv_list : a list of strings, string of csv path
    useful_stations : dict of {aq_station : meo_station}
    '''

    # csv_list = ["./KDD_CUP_2018/Beijing/grid_meo/Beijing_historical_meo_grid.csv", 
    #             "./KDD_CUP_2018/Beijing/grid_meo/new.csv"]

    path_to_bj_meo = "./KDD_CUP_2018/Beijing/grid_meo/"
    bj_csv_list  = os.listdir(path_to_bj_meo)

    bj_meo_datas = []

    for csv in bj_csv_list :
        if csv != '.DS_Store' and not csv.startswith("._"):
            path_to_file = path_to_bj_meo + csv
            print(path_to_file)
            bj_meo_data = pd.read_csv(path_to_file)
            print(bj_meo_data.columns)

            # 去掉多余信息
            if "longitude" in bj_meo_data.columns :
                bj_meo_data.drop("longitude", axis=1, inplace=True)
            if "latitude" in bj_meo_data.columns :    
                bj_meo_data.drop("latitude", axis=1, inplace=True)
            if "id" in bj_meo_data.columns :
                bj_meo_data.drop("id", axis=1, inplace=True)
            if "weather" in bj_meo_data.columns :
                bj_meo_data.drop("weather", axis=1, inplace=True)
            
            name_pairs = {}
            if "station_id" in bj_meo_data.columns :
                name_pairs["station_id"] = "stationName"
            if "time" in bj_meo_data.columns :
                name_pairs["time"] = "utc_time"
            if "wind_speed/kph" in bj_meo_data.columns :
                name_pairs["wind_speed/kph"] = "wind_speed"
            
            bj_meo_data.rename(index=str, columns=name_pairs, inplace=True)
            bj_meo_datas.append(bj_meo_data)

    meo_dataset = pd.concat(bj_meo_datas, ignore_index=True)

    bj_grid_meo_dataset, stations, bj_meo_stations = load_grid_meo_data(meo_dataset, useful_stations)

    return bj_grid_meo_dataset, stations, bj_meo_stations


def load_ld_grid_meo_data(useful_stations):
    '''
    csv_list : a list of strings, string of csv path
    useful_stations : dict of {aq_station : meo_station}
    '''

    # csv_list = ["./KDD_CUP_2018/London/grid_meo/London_historical_meo_grid.csv"]

    path_to_ld_meo = "./KDD_CUP_2018/London/grid_meo/"
    ld_csv_list  = os.listdir(path_to_ld_meo)

    ld_meo_datas = []

    for csv in ld_csv_list :
        if csv != '.DS_Store' and not csv.startswith("._"):
            path_to_file = path_to_bj_meo + csv
            print(path_to_file)
            ld_meo_data = pd.read_csv(path_to_file)
            print(ld_meo_data.columns)

            # 去掉多余信息
            if "longitude" in ld_meo_data.columns :
                ld_meo_data.drop("longitude", axis=1, inplace=True)
            if "latitude" in ld_meo_data.columns :    
                ld_meo_data.drop("latitude", axis=1, inplace=True)
            if "id" in ld_meo_data.columns :
                ld_meo_data.drop("id", axis=1, inplace=True)
            if "weather" in ld_meo_data.columns :
                ld_meo_data.drop("weather", axis=1, inplace=True)
            
            name_pairs = {}
            if "station_id" in ld_meo_data.columns :
                name_pairs["station_id"] = "stationName"
            if "time" in ld_meo_data.columns :
                name_pairs["time"] = "utc_time"
            if "wind_speed/kph" in ld_meo_data.columns :
                name_pairs["wind_speed/kph"] = "wind_speed"
            
            ld_meo_data.rename(index=str, columns=name_pairs, inplace=True)
            ld_meo_datas.append(ld_meo_data)

    meo_dataset = pd.concat(ld_meo_datas, ignore_index=True)

    ld_grid_meo_dataset, stations, ld_meo_stations = load_grid_meo_data(meo_dataset, useful_stations)

    return ld_grid_meo_dataset, stations, ld_meo_stations


def load_grid_meo_data(meo_df, useful_stations):
    '''
    useful_stations : dict of {aq_station : meo_station}
    '''

    meo_dataset = meo_df

    # turn date from string type to datetime type
    meo_dataset["time"] = pd.to_datetime(meo_dataset['utc_time'])
    meo_dataset.set_index("time", inplace=True)
    meo_dataset.drop("utc_time", axis=1, inplace=True)

    # names of all stations
    stations = set(meo_dataset['stationName'])

    # a dict of station aq, Beijing
    meo_stations = {}

    for aq_station_name, meo_station_name in useful_stations.items() :

        if meo_station_name in stations :
            meo_station = meo_dataset[meo_dataset["stationName"]==meo_station_name].copy()
            meo_station.drop("stationName", axis=1, inplace=True)

            # rename
            original_names = meo_station.columns.values.tolist()
            names_dict = {original_name : aq_station_name+"_"+original_name for original_name in original_names}
            meo_station_renamed = meo_station.rename(index=str, columns=names_dict)
            

            meo_stations[aq_station_name] = meo_station_renamed        


    return meo_dataset, stations, meo_stations














def get_station_locations(stations_df):
    '''
    Get all the locations of stations in stations_df.
    Agrs : 
        stations_df : a dataframe of all station data.
    Return : 
        A list of (station_name, (longitude, latitude))
    '''
    
    locations = []
    station_names = []
    
    if 'station_id' in stations_df.columns:
        station_column_name = 'station_id'
    elif 'stationName' in stations_df.columns:
        station_column_name = 'stationName'
    else :
        print("Can not find station name!")
    
    for j in stations_df.index:
        station_name = stations_df[station_column_name][j]
        if station_name not in station_names:
            station_names.append(station_name)
            longitude = stations_df['longitude'][j]
            latitude = stations_df['latitude'][j]
            location = (longitude, latitude)
            # station_name = stations_df[station_column_name][j]
            locations.append((station_name, location))
    
    return locations


def get_location_lists(locations):
    '''
    Get location list from locations.
    Args : 
        A list with element shape (station_name, (longitude, latitude)).
    Return : 
        Two lists of longitudes and latitudes.
    '''
    longitudes = []
    latitudes = []
    
    for i in range(len(locations)):
        _, (longitude, latitude) = locations[i]
        longitudes.append(longitude)
        latitudes.append(latitude)
        
    return longitudes, latitudes


def find_nearst_meo_station_name(aq_location, meo_locations):
    '''
    From meo stations ans grid meos stations, find the nearest meo station of aq station.
    Args :
        aq_location : an aq station information of (station_name, (longitude, latitude))
        meo_locations : meo information, list of ((station_name, (longitude, latitude)))
    '''
    nearest_station_name = ""
    nearest_distance = 1e10
    
    aq_station_longitude = aq_location[1][0]
    aq_station_latitude = aq_location[1][1]
    
    for station_name, (longitude, latitude) in meo_locations:
        dis = np.sqrt((longitude-aq_station_longitude)**2 + (latitude-aq_station_latitude)**2)
        if dis < nearest_distance:
            nearest_distance = dis
            nearest_station_name = station_name
    
    return nearest_station_name


def get_related_meo_dfs(aq_station_nearest_meo_station, bj_meo_all, bj_grid_meo_all):
    '''
    Get a dict with aq_station_name as key and nearest meo_station meo data as value.
    Args :
        aq_station_nearest_meo_station = {aq_station_name : meo_station_name}
        bj_meo_all is Beijing meo dataframe.
        grid_bj_meo_all is Beijing grid meo dataframe.
    Returns:
        related_meo_dfs = {aq_station_name : meo_station_data_df}
    '''
    related_meo_dfs = {}
    
    bj_meo_all_names = set(bj_meo_all["station_id"].values)
    grid_bj_meo_all_names = set(bj_grid_meo_all["stationName"].values)

    for aq_station, meo_station in aq_station_nearest_meo_station.items():
        if meo_station in bj_meo_all_names:
            related_meo_dfs[aq_station] = bj_meo_all[bj_meo_all['station_id'] == meo_station]
        elif meo_station in grid_bj_meo_all_names:
            related_meo_dfs[aq_station] = bj_grid_meo_all[bj_grid_meo_all['stationName'] == meo_station]
        else :
            print("meo station name not found.")
    
    return related_meo_dfs



# def load_bj_grid_meo_data(useful_stations):
#     '''
#     csv_list : a list of strings, string of csv path
#     useful_stations : dict of {aq_station : meo_station}
#     '''


#     # 网格气象数据
#     bj_grid_meo_dataset_1 = pd.read_csv("./KDD_CUP_2018/Beijing/grid_meo/Beijing_historical_meo_grid.csv")
#     # API 下载数据
#     bj_grid_meo_dataset_2 = pd.read_csv("./KDD_CUP_2018/Beijing/grid_meo/new.csv")

#     # 去掉位置信息
#     bj_grid_meo_dataset_1.drop("longitude", axis=1, inplace=True)
#     bj_grid_meo_dataset_1.drop("latitude", axis=1, inplace=True)

 
#     bj_grid_meo_dataset = pd.concat([bj_grid_meo_dataset_1, bj_grid_meo_dataset_2], ignore_index=True)

#     # turn date from string type to datetime type
#     bj_grid_meo_dataset["time"] = pd.to_datetime(bj_grid_meo_dataset['utc_time'])
#     bj_grid_meo_dataset.set_index("time", inplace=True)
#     bj_grid_meo_dataset.drop("utc_time", axis=1, inplace=True)

#     # names of all stations
#     stations = set(bj_grid_meo_dataset['stationName'])

#     # a dict of station aq, Beijing
#     bj_meo_stations = {}

#     for aq_station, meo_station in useful_stations.items() :

#         if meo_station in stations :
#             bj_meo_station = bj_grid_meo_dataset[bj_grid_meo_dataset["stationName"]==meo_station].copy()
#             bj_meo_station.drop("stationName", axis=1, inplace=True)

#             # rename
#             original_names = bj_meo_station.columns.values.tolist()
#             names_dict = {original_name : aq_station+"_"+original_name for original_name in original_names}
#             bj_meo_station_renamed = bj_meo_station.rename(index=str, columns=names_dict)
            

#             bj_meo_stations[aq_station] = bj_meo_station_renamed        


#     return bj_grid_meo_dataset, stations, bj_meo_stations

