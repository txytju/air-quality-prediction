# coding: utf-8

# Using pandas to process data
from collections import Counter
import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from utils.data_util import load_ld_aq_data, load_bj_aq_data

stations_to_predict = ['BL0','CD1','CD9','GN0','GN3','GR4','GR9','HV1','KF1','LW2','MY7','ST5','TH4'] # 13个
other_stations = ['BX1', 'BX9', 'CR8', 'CT2', 'CT3', 'GB0', 'HR1', 'KC1', 'LH0', 'RB7', 'TD5']        # 11个
features = ['NO2 (ug/m3)', 'PM10 (ug/m3)', 'PM2.5 (ug/m3)']

all_features = []
for station in stations_to_predict :
    for feature in features:
        station_feature = station + "_" + feature
        all_features.append(station_feature)


ld_aq_data, stations, ld_aq_stations, ld_aq_stations_merged = load_ld_aq_data()

print("最早的日期：", ld_aq_stations_merged.index.min())
print("最晚的日期：", ld_aq_stations_merged.index.max())

df_merged = ld_aq_stations_merged

# ### 3. 缺失值的分析
# 3.1 某个小时某个站点数据缺失
# - 在没有全部缺失的小时处，某个站点会在某个小时出现全部数据缺失的情况。这种情况下，使用距离该站最近的站的数据对其进行补全。
# - 或者某个站点的某个值缺失，此时使用相邻站点的数据补全

# from utils.weather_data_util import get_station_locations, get_location_lists



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


# #### 3.3 个别缺失的处理

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
        else :
            return 0


for index in df_merged.index :
    row = df_merged.loc[index]
    for feature in row.index :
        # print(feature)
        if pd.isnull(row[feature]) :
            
            elements = feature.split("_")                  # feature example： KC1_NO2 (ug/m3)
            station_name = elements[0]                     # KC1
            feature_name = elements[1]                   # NO2 (ug/m3)
            row[feature] = get_estimated_value(station_name, feature_name, near_stations, row)

df_merged = df_merged[all_features]

df_merged.to_csv("test/ld_aq_data.csv")


# 4. 数据归一化
describe = df_merged.describe()
describe.to_csv("test/ld_aq_describe.csv")

df_norm = (df_merged - describe.loc['mean']) / describe.loc['std']
df_norm.to_csv("test/ld_aq_norm_data.csv")



