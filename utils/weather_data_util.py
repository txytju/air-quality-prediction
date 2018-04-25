import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt

def load_bj_grid_meo_data():

    # 网格气象数据
    bj_grid_meo_dataset_1 = pd.read_csv("./KDD_CUP_2018/Beijing/grid_meo/Beijing_historical_meo_grid.csv")
    # API 下载数据
    bj_grid_meo_dataset_2 = pd.read_csv("./KDD_CUP_2018/Beijing/grid_meo/new.csv")

    bj_grid_meo_dataset = pd.concat([bj_grid_meo_dataset_1, bj_grid_meo_dataset_2], ignore_index=True)

    # turn date from string type to datetime type
    bj_grid_meo_dataset["time"] = pd.to_datetime(bj_grid_meo_dataset['utc_time'])
    bj_grid_meo_dataset.set_index("time", inplace=True)
    bj_grid_meo_dataset.drop("utc_time", axis=1, inplace=True)

    # names of all stations
    stations = set(bj_grid_meo_dataset['stationName'])

    # a dict of station aq, Beijing
    bj_meo_stations = {}
    for station in stations:
        bj_meo_station = bj_grid_meo_dataset[bj_grid_meo_dataset["stationName"]==station].copy()
        bj_meo_station.drop("stationName", axis=1, inplace=True)

        # rename
        original_names = bj_meo_station.columns.values.tolist()
        names_dict = {original_name : station+"_"+original_name for original_name in original_names}
        bj_meo_station_renamed = bj_meo_station.rename(index=str, columns=names_dict)
        
        length = bj_meo_station_renamed.shape[0]
        order = range(length)
        bj_meo_station_renamed['order'] = pd.Series(order, index=bj_meo_station_renamed.index)


        bj_meo_stations[station] = bj_meo_station_renamed


    return bj_grid_meo_dataset, stations, bj_meo_stations








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





