## 主程序

# 数据的自动获取


# 数据预处理

from collections import Counter
import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from utils.data_util import load_bj_aq_data, generate_model_data
from utils.weather_data_util import get_station_locations, get_location_lists

bj_aq_data, stations, bj_aq_stations, bj_aq_stations_merged = load_bj_aq_data()

print("最早的日期：", bj_aq_stations_merged.index.min())
print("最晚的日期：", bj_aq_stations_merged.index.max())

df_merged = bj_aq_stations_merged
df_merged["time"] = df_merged.index
df_merged.set_index("order", inplace=True)
print("重复值去除之前，共有数据数量", df_merged.shape[0])

used_times = []
for index in df_merged.index :
    time = df_merged.loc[index]["time"]
    if time not in used_times :
        used_times.append(time)
    else : 
        df_merged.drop([index], inplace=True)

print("重复值去除之后，共有数据数量", df_merged.shape[0])

df_merged.set_index("time", inplace=True)

min_time = df_merged.index.min()
max_time = df_merged.index.max()

min_time = datetime.datetime.strptime(min_time, '%Y-%m-%d %H:%M:%S')
max_time = datetime.datetime.strptime(max_time, '%Y-%m-%d %H:%M:%S')
delta_all = max_time - min_time
print("在空气质量数据时间段内，总共应该有 %d 个小时节点。" %(delta_all.total_seconds()/3600 + 1))
print("实际的时间节点数是 %d" %(df_merged.shape[0]))
print("缺失时间节点数量是 %d" %(10898-10113))


delta = datetime.timedelta(hours=1)
time = min_time
missing_hours = []
missing_hours_str = []


while time <=  max_time :
    if datetime.date.strftime(time, '%Y-%m-%d %H:%M:%S') not in df_merged.index :
        missing_hours.append(time)
        missing_hours_str.append(datetime.date.strftime(time, '%Y-%m-%d %H:%M:%S'))
    time += delta


print("整小时的缺失共计 : ", len(missing_hours))


aq_station_locations = pd.read_excel("./KDD_CUP_2018/Beijing/location/Beijing_AirQuality_Stations_locations.xlsx", sheet_name=1)


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
        else :
            return 0



for index in df_merged.index :
    row = df_merged.loc[index]
    for feature in row.index :
        # print(feature)
        if pd.isnull(row[feature]) :
            elements = feature.split("_")                  # feature example： nansanhuan_aq_PM2.5
            station_name = elements[0] + "_" + elements[1] # nansanhuan_aq
            feature_name = elements[2]                     # PM2.5
            row[feature] = get_estimated_value(station_name, feature_name, near_stations, row)



# 现在数据中没有缺失值了 :)
print(pd.isnull(df_merged).any().any())

print(df_merged.shape)



# 将 missing hours 分成两类 : keep_hours and drop_hours
keep_hours = []
drop_hours = []

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


# keep 180 hours of 785 missing hours
print(len(drop_hours), len(keep_hours), len(missing_hours_str))
print(df_merged.shape)
print(pd.isnull(df_merged).any().any())


nan_series = pd.Series({key:np.nan for key in df_merged.columns})

for hour in drop_hours:
    df_merged.loc[hour] = nan_series


df_merged.sort_index(inplace=True)

df_merged.to_csv("test/bj_aq_data.csv")



describe = df_merged.describe()
describe.to_csv("test/bj_aq_describe.csv")


df_norm = (df_merged - describe.loc['mean']) / describe.loc['std']
df_norm.to_csv("test/bj_aq_norm_data.csv")














