# coding: utf-8

# Using pandas to process data
from collections import Counter
import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from utils.data_util import load_bj_aq_data, load_ld_aq_data

from utils.weather_data_util import get_station_locations, get_location_lists


def aq_data_preprocess(city="bj"):

    # 1. 数据载入
    if city == "bj" :
        aq_data, stations, aq_stations, aq_stations_merged = load_bj_aq_data()
    elif city == "ld" :
        aq_data, stations, aq_stations, aq_stations_merged = load_ld_aq_data()

    # 2. 重复日期的值除去
    df_merged = aq_stations_merged
    df_merged["time"] = df_merged.index
    df_merged.set_index("order", inplace=True)

    used_times = []
    for index in df_merged.index :
        time = df_merged.loc[index]["time"]
        if time not in used_times :
            used_times.append(time)
        else : 
            df_merged.drop([index], inplace=True)

    df_merged.set_index("time", inplace=True)


    # 3. 缺失值的分析
    min_time = df_merged.index.min()
    max_time = df_merged.index.max()

    min_time = datetime.datetime.strptime(min_time, '%Y-%m-%d %H:%M:%S')
    max_time = datetime.datetime.strptime(max_time, '%Y-%m-%d %H:%M:%S')
    delta_all = max_time - min_time

    hours_should = delta_all.total_seconds()/3600 + 1


    # 3.1 整小时缺失

    delta = datetime.timedelta(hours=1)
    time = min_time
    missing_hours = []
    missing_hours_str = []


    while time <=  max_time :
        if datetime.date.strftime(time, '%Y-%m-%d %H:%M:%S') not in df_merged.index :
            missing_hours.append(time)
            missing_hours_str.append(datetime.date.strftime(time, '%Y-%m-%d %H:%M:%S'))
        time += delta


    # 3.2 某个小时某个站点数据缺失
    if city == "bj" :
        aq_station_locations = pd.read_excel("./KDD_CUP_2018/Beijing/location/Beijing_AirQuality_Stations_locations.xlsx", sheet_name=0)
    elif city == "ld" :
        aq_station_locations = pd.read_csv("./KDD_CUP_2018/London/location/London_AirQuality_Stations.csv")
        aq_station_locations = aq_station_locations[["Unnamed: 0", "Latitude", "Longitude"]]
        aq_station_locations.rename(index=str, columns={"Unnamed: 0":"stationName", "Latitude":"latitude", "Longitude":"longitude"}, inplace=True)

    # 对于一个空气质量站点，将其他站点按照距该站点距离的大小关系排列，并保存成列表

    for index_t in aq_station_locations.index:
        row_t = aq_station_locations.loc[index_t]
        # location of target station
        long_t = row_t["longitude"]
        lati_t = row_t["latitude"]
        # column name
        station_name = row_t["stationName"]
        
        # add a new column to df
        all_dis = []
        for index in aq_station_locations.index:
            row = aq_station_locations.loc[index]
            long = row['longitude']
            lati = row['latitude']
            dis = np.sqrt((long-long_t)**2 + (lati-lati_t)**2)
            all_dis.append(dis)
        
        aq_station_locations[station_name] = all_dis

    # 以每一个站的名字为 key，以其他站的名字组成的列表为 value list，列表中从前向后距离越来越远
    near_stations = {}
    for index_t in aq_station_locations.index:
        target_station_name = aq_station_locations.loc[index_t]['stationName']
        ordered_stations_names = aq_station_locations.sort_values(by=target_station_name)['stationName'].values[1:]
        near_stations[target_station_name] = ordered_stations_names


    def get_estimated_value(station_name, feature_name, near_stations, row):
        '''
        为 feature 寻找合理的缺失值的替代。
        Args:
            near_stations : a dict of {station : near stations}
        '''   
        near_stations = near_stations[station_name]    # A list of nearest stations
        for station in near_stations :                 # 在最近的站中依次寻找非缺失值
            feature = station + "_" +feature_name
            if not pd.isnull(row[feature]):
                return row[feature]
            
        return 0


    for index in df_merged.index :
        row = df_merged.loc[index].copy()
        for feature in row.index :
            # print(feature)
            if pd.isnull(row[feature]) :
                elements = feature.split("_")                  
                if city == "bj" :                                  # feature example： nansanhuan_aq_PM2.5
                    station_name = elements[0] + "_" + elements[1] # nansanhuan_aq
                    feature_name = elements[2]                     # PM2.5
                elif city == "ld" :                                # feature example： KC1_NO2 (ug/m3)
                    station_name = elements[0]                     # KC1
                    feature_name = elements[1]                     # NO2 (ug/m3)
                row[feature] = get_estimated_value(station_name, feature_name, near_stations, row)
        df_merged.loc[index] = row

    assert (pd.isnull(df_merged).any().any()) == False, "数据中还有缺失值(局部处理后)"

    # London 并不是每个站点都有用
    if city == "ld" :
        
        stations_to_predict = ['BL0','CD1','CD9','GN0','GN3','GR4','GR9','HV1','KF1','LW2','MY7','ST5','TH4'] # 13个
        other_stations = ['BX1', 'BX9', 'CR8', 'CT2', 'CT3', 'GB0', 'HR1', 'KC1', 'LH0', 'RB7', 'TD5']        # 11个
        features = ['NO2 (ug/m3)', 'PM10 (ug/m3)', 'PM2.5 (ug/m3)']

        all_features = []
        for station in stations_to_predict :
            for feature in features:
                station_feature = station + "_" + feature
                all_features.append(station_feature)

        df_merged = df_merged[all_features]
    # 3.3 整小时的缺失的处理
    keep_hours = []
    drop_hours = []


    # 先对小于5小时的进行填充
    delta = datetime.timedelta(hours=1)

    for hour in missing_hours_str : 
        
        time = datetime.datetime.strptime(hour, '%Y-%m-%d %H:%M:%S')
        
        # 前边第几个是非空的
        found_for = False
        i = 0
        while not found_for :
            i += 1
            for_time = time - i * delta
            for_time_str = datetime.date.strftime(for_time, '%Y-%m-%d %H:%M:%S')
            if for_time_str in df_merged.index :
                for_row = df_merged.loc[for_time_str]
                for_step = i
                found_for = True
                
                
        # 后边第几个是非空的
        found_back = False
        j = 0
        while not found_back :
            j += 1
            back_time = time + j * delta
            back_time_str = datetime.date.strftime(back_time, '%Y-%m-%d %H:%M:%S')
            if back_time_str in df_merged.index :
                back_row = df_merged.loc[back_time_str]
                back_step = j
                found_back = True
        
        # print(for_step, back_step)
        all_steps = for_step + back_step
        if all_steps > 5 :
            drop_hours.append(hour)
        else : 
            keep_hours.append(hour)
            # 插值
            delata_values = back_row - for_row
            df_merged.loc[hour] = for_row + (for_step/all_steps) * delata_values        



    assert pd.isnull(df_merged).any().any() == False, "数据中还有缺失值(整体处理后)"


    # 再对超过5小时的填充 NAN
    nan_series = pd.Series({key:np.nan for key in df_merged.columns})

    for hour in drop_hours:
        df_merged.loc[hour] = nan_series

    df_merged.sort_index(inplace=True)

    assert df_merged.shape[0] == hours_should , "填充完的长度和应有的长度不一致"

    # 3.4 数据存储

    df_merged.to_csv("test/%s_aq_data.csv" %(city))


    # ### 4. 数据归一化

    describe = df_merged.describe()
    describe.to_csv("test/%s_aq_describe.csv" %(city))

    df_norm = (df_merged - describe.loc['mean']) / describe.loc['std']
    df_norm.to_csv("test/%s_aq_norm_data.csv" %(city))

    print("完成对 %s 空气质量数据的预处理！" %(city))
