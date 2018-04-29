# coding: utf-8

import numpy as np
import datetime
import pandas as pd
from utils.weather_data_util import load_ld_grid_meo_data

near_stations = {'BL0': 'london_grid_409',
 'BX1': 'london_grid_472',
 'BX9': 'london_grid_472',
 'CD1': 'london_grid_388',
 'CD9': 'london_grid_409',
 'CR8': 'london_grid_408',
 'CT2': 'london_grid_409',
 'CT3': 'london_grid_409',
 'GB0': 'london_grid_451',
 'GN0': 'london_grid_451',
 'GN3': 'london_grid_451',
 'GR4': 'london_grid_451',
 'GR9': 'london_grid_430',
 'HR1': 'london_grid_368',
 'HV1': 'london_grid_472',
 'KC1': 'london_grid_388',
 'KF1': 'london_grid_388',
 'LH0': 'london_grid_346',
 'LW2': 'london_grid_430',
 'MY7': 'london_grid_388',
 'RB7': 'london_grid_452',
 'ST5': 'london_grid_408',
 'TD5': 'london_grid_366',
 'TH4': 'london_grid_430'}


# ld_meo_stations : meo_station as key and df as value 
ld_grid_meo_dataset, stations, ld_meo_stations = load_ld_grid_meo_data(near_stations)

print(ld_meo_stations['BX1'].index.min(), ld_meo_stations['BX1'].index.max())




# #### 3.4 风向缺失值处理
# - 暂时使用 0 替换缺失的风向

# In[16]:


for station in ld_meo_stations.keys():
    df = ld_meo_stations[station].copy()
    df.replace(999017,0, inplace=True)
    ld_meo_stations[station] = df


# #### 3.5 拼成整表，并保存


ld_meo_stations_merged = pd.concat(list(ld_meo_stations.values()), axis=1)




stations_to_predict = ['BL0','CD1','CD9','GN0','GN3','GR4','GR9','HV1','KF1','LW2','MY7','ST5','TH4']




all_features = []
for column_name in ld_meo_stations_merged.columns : 
    for station in stations_to_predict :
        if station in column_name :
            all_features.append(column_name)



ld_meo_stations_merged = ld_meo_stations_merged[all_features]

ld_meo_stations_merged.shape


ld_meo_stations_merged.sort_index(inplace=True)


ld_meo_stations_merged.to_csv("test/ld_meo_data.csv")


# 需要将 `ld_meo_data.csv`左上角位置的空格补上 "time" 

# ### 4 数据归一化

describe = ld_meo_stations_merged.describe()
describe.to_csv("test/ld_meo_describe.csv")

df_norm = (ld_meo_stations_merged - describe.loc['mean']) / describe.loc['std']
df_norm.to_csv("test/ld_meo_norm_data.csv")

