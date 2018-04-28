# coding: utf-8

# Using pandas to process data
from collections import Counter
import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from utils.data_util import load_bj_aq_data, generate_model_data
from utils.weather_data_util import get_station_locations, get_location_lists

# ### 1. 数据载入
bj_aq_data, stations, bj_aq_stations, bj_aq_stations_merged = load_bj_aq_data()

# print("最早的日期：", bj_aq_stations_merged.index.min())
# print("最晚的日期：", bj_aq_stations_merged.index.max())


# ### 2. 重复日期的值除去

# - 虽然数据中存在着很多的缺失值，但是在一些时间点上数据是重复的，因此首先去除重复的时间点。
# - 思路是：
#     1. 首先让数字成为 df index
#     2. 遍历 index ，找出对应的时间的重复值，删除后出现的重复时间

df_merged = bj_aq_stations_merged
df_merged["time"] = df_merged.index
df_merged.set_index("order", inplace=True)
# print("重复值去除之前，共有数据数量", df_merged.shape[0])

used_times = []
for index in df_merged.index :
    time = df_merged.loc[index]["time"]
    if time not in used_times :
        used_times.append(time)
    else : 
        df_merged.drop([index], inplace=True)

# print("重复值去除之后，共有数据数量", df_merged.shape[0])

df_merged.set_index("time", inplace=True)


# ### 3. 缺失值的分析

# - 缺失值分析包括
#     - 整小时的缺失：哪些时间节点上，所有站点的所有特征都缺失了
#     - 某个小时的某个站点的所有数据缺失
#     - 某个小时的某个站点的某个数据缺失
# - 对于上述第一种情况，如果某天的缺失数据超过5个小时，就放弃使用该天的数据，如果没有超过5个小时，使用插值的方式对数据进行填充。
# - 对于后两种情况，使用距离该站的有数据的最近一个站的数据，直接作为该站的数据。


min_time = df_merged.index.min()
max_time = df_merged.index.max()

min_time = datetime.datetime.strptime(min_time, '%Y-%m-%d %H:%M:%S')
max_time = datetime.datetime.strptime(max_time, '%Y-%m-%d %H:%M:%S')
delta_all = max_time - min_time
# print("在空气质量数据时间段内，总共应该有 %d 个小时节点。" %(delta_all.total_seconds()/3600 + 1))
# print("实际的时间节点数是 %d" %(df_merged.shape[0]))



# #### 3.1 整小时缺失
# 统计哪些时间节点发生了整小时缺失的情况


delta = datetime.timedelta(hours=1)
time = min_time
missing_hours = []
missing_hours_str = []


while time <=  max_time :
    if datetime.date.strftime(time, '%Y-%m-%d %H:%M:%S') not in df_merged.index :
        missing_hours.append(time)
        missing_hours_str.append(datetime.date.strftime(time, '%Y-%m-%d %H:%M:%S'))
    time += delta

# print("整小时的缺失共计 : ", len(missing_hours))


# #### 3.2 某个小时某个站点数据缺失
# - 在没有全部缺失的小时处，某个站点会在某个小时出现全部数据缺失的情况。这种情况下，使用距离该站最近的站的数据对其进行补全。
# - 或者某个站点的某个值缺失，此时使用相邻站点的数据补全

aq_station_locations = pd.read_excel("./KDD_CUP_2018/Beijing/location/Beijing_AirQuality_Stations_locations.xlsx", sheet_name=0)


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
            elements = feature.split("_")                  # feature example： nansanhuan_aq_PM2.5
            station_name = elements[0] + "_" + elements[1] # nansanhuan_aq
            feature_name = elements[2]                     # PM2.5
            row[feature] = get_estimated_value(station_name, feature_name, near_stations, row)




# 现在数据中没有缺失值了 :)
assert (pd.isnull(df_merged).any().any()) == False, "数据中还有缺失值(局部处理后)"


# #### 3.4 整小时的缺失的处理

# 统计缺失小时的连续值
# - 如果一个缺失小时在一个长度小于等于5小时的缺失时段内，就进行补全
# - 如果超过5小时，就舍弃，将全部值补成 NAN，**这也是整个表中唯一可能出现 NAN 的情况**


# 将 missing hours 分成两类 : keep_hours and drop_hours
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



# 依然 没有 Nan，棒！
assert pd.isnull(df_merged).any().any() == False, "数据中还有缺失值(整体处理后)"


# 再对超过5小时的填充 NAN
nan_series = pd.Series({key:np.nan for key in df_merged.columns})

for hour in drop_hours:
    df_merged.loc[hour] = nan_series


df_merged.sort_index(inplace=True)


# 11458 和应有的数量一致 :)
assert df_merged.shape == delta_all.total_seconds()/3600 + 1 , "填充完的长度和应有的长度不一致"


df_merged.to_csv("test/bj_aq_data.csv")


# ### 4. 数据归一化

describe = df_merged.describe()
describe.to_csv("test/bj_aq_describe.csv")

df_norm = (df_merged - describe.loc['mean']) / describe.loc['std']
df_norm.to_csv("test/bj_aq_norm_data.csv")
