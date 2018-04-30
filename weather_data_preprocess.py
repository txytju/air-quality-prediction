# coding: utf-8

import numpy as np
import datetime
import pandas as pd
from utils.weather_data_util import load_bj_grid_meo_data, load_ld_grid_meo_data

bj_near_stations = {'aotizhongxin_aq': 'beijing_grid_304',
 'badaling_aq': 'beijing_grid_224',
 'beibuxinqu_aq': 'beijing_grid_263',
 'daxing_aq': 'beijing_grid_301',
 'dingling_aq': 'beijing_grid_265',
 'donggaocun_aq': 'beijing_grid_452',
 'dongsi_aq': 'beijing_grid_303',
 'dongsihuan_aq': 'beijing_grid_324',
 'fangshan_aq': 'beijing_grid_238',
 'fengtaihuayuan_aq': 'beijing_grid_282',
 'guanyuan_aq': 'beijing_grid_282',
 'gucheng_aq': 'beijing_grid_261',
 'huairou_aq': 'beijing_grid_349',
 'liulihe_aq': 'beijing_grid_216',
 'mentougou_aq': 'beijing_grid_240',
 'miyun_aq': 'beijing_grid_392',
 'miyunshuiku_aq': 'beijing_grid_414',
 'nansanhuan_aq': 'beijing_grid_303',
 'nongzhanguan_aq': 'beijing_grid_324',
 'pingchang_aq': 'beijing_grid_264',
 'pinggu_aq': 'beijing_grid_452',
 'qianmen_aq': 'beijing_grid_303',
 'shunyi_aq': 'beijing_grid_368',
 'tiantan_aq': 'beijing_grid_303',
 'tongzhou_aq': 'beijing_grid_366',
 'wanliu_aq': 'beijing_grid_283',
 'wanshouxigong_aq': 'beijing_grid_303',
 'xizhimenbei_aq': 'beijing_grid_283',
 'yanqin_aq': 'beijing_grid_225',
 'yizhuang_aq': 'beijing_grid_323',
 'yongdingmennei_aq': 'beijing_grid_303',
 'yongledian_aq': 'beijing_grid_385',
 'yufa_aq': 'beijing_grid_278',
 'yungang_aq': 'beijing_grid_239',
 'zhiwuyuan_aq': 'beijing_grid_262'}

ld_near_stations = {'BL0': 'london_grid_409',
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


def meo_data_preprocess(city="bj"):

    # 1. 数据载入
    if city == "bj" :
        grid_meo_dataset, stations, meo_stations = load_bj_grid_meo_data(bj_near_stations)
    elif city == "ld" :
        grid_meo_dataset, stations, meo_stations = load_ld_grid_meo_data(ld_near_stations)

    # ### 2. 重复值分析
    for station in meo_stations.keys() :
        
        df = meo_stations[station].copy()
        length = df.shape[0]
        order = range(length)
        df['order'] = pd.Series(order, index=df.index)    
        
        
        df["time"] = df.index
        df.set_index("order", inplace=True)
        
        length_1 = df.shape[0]
        # print("重复值去除之前，共有数据数量", df.shape[0])
        
        used_times = []
        for index in df.index :
            time = df.loc[index]["time"]
            if time not in used_times :
                used_times.append(time)
            else : 
                df.drop([index], inplace=True)

        length_2 = df.shape[0]
        delta = length_1 - length_2
        # print("重复值去除之后，共有数据数量", df.shape[0])
        # print("%s 重复数量 : %d" %(station, delta))
        
        df.set_index("time", inplace=True)
        meo_stations[station] = df


    # 3. 缺失值分析

    # 3.2 整体缺失

    # 统计缺失小时的连续值
    # - 如果一个缺失小时在一个长度小于等于5小时的缺失时段内，就进行补全
    # - 如果超过5小时，就舍弃，将全部值补成 NAN，**这也是整个表中唯一可能出现 NAN 的情况**

    for station in meo_stations.keys() :
        df = meo_stations[station].copy()
        
        min_time = df.index.min()
        max_time = df.index.max()

        min_time = datetime.datetime.strptime(min_time, '%Y-%m-%d %H:%M:%S')
        max_time = datetime.datetime.strptime(max_time, '%Y-%m-%d %H:%M:%S')
        delta_all = max_time - min_time
        
        all_length = delta_all.total_seconds()/3600 + 1
        real_length = df.shape[0]
        # print("在空气质量数据时间段内，总共应该有 %d 个小时节点。" %(all_length))
        # print("实际的时间节点数是 ", real_length)
        # print("%s 缺失时间节点数量是 %d" %(station, all_length-real_length))


    # #### 3.3 整体缺失补充

    for station in meo_stations.keys() :
        df = meo_stations[station].copy()
        # print(station, df.shape)



    delta = datetime.timedelta(hours=1)

    for station in meo_stations.keys() :
        df = meo_stations[station].copy()
        nan_series = pd.Series({key:np.nan for key in df.columns})
        
        min_time = df.index.min()
        max_time = df.index.max()

        min_time = datetime.datetime.strptime(min_time, '%Y-%m-%d %H:%M:%S')
        max_time = datetime.datetime.strptime(max_time, '%Y-%m-%d %H:%M:%S')

        time = min_time
        
        while time <=  max_time :
            
            time_str = datetime.date.strftime(time, '%Y-%m-%d %H:%M:%S')
            if time_str not in df.index :
                
                # 前边第几个是非空的
                found_for = False
                i = 0
                while not found_for :
                    i += 1
                    for_time = time - i * delta
                    for_time_str = datetime.date.strftime(for_time, '%Y-%m-%d %H:%M:%S')
                    if for_time_str in df.index :
                        for_row = df.loc[for_time_str]
                        for_step = i
                        found_for = True

                # 后边第几个是非空的
                found_back = False
                j = 0
                while not found_back :
                    j += 1
                    back_time = time + j * delta
                    back_time_str = datetime.date.strftime(back_time, '%Y-%m-%d %H:%M:%S')
                    if back_time_str in df.index :
                        back_row = df.loc[back_time_str]
                        back_step = j
                        found_back = True
            
                all_steps = for_step + back_step
            
                if all_steps <= 5 :
                    delata_values = back_row - for_row
                    df.loc[time_str] = for_row + (for_step/all_steps) * delata_values
                else :
                    df.loc[time_str] = nan_series
            
            time += delta
        meo_stations[station] = df
        
        # print("%s : length of data is %d" %(station, df.shape[0]))


    # #### 3.4 风向缺失值处理


    for station in meo_stations.keys():
        df = meo_stations[station].copy()
        df.replace(999017,0, inplace=True)
        meo_stations[station] = df


    # #### 3.5 拼成整表，并保存


    meo_stations_merged = pd.concat(list(meo_stations.values()), axis=1)
    meo_stations_merged.sort_index(inplace=True)
    print("将要保存的天气数据的尺寸是　",meo_stations_merged.shape)

    meo_stations_merged.to_csv("test/%s_meo_data.csv" %(city))


    # 需要将 `bj_meo_data.csv`左上角位置的空格补上 "time" 

    # ### 4 数据归一化


    describe = meo_stations_merged.describe()
    describe.to_csv("test/%s_meo_describe.csv" %(city))


    df_norm = (meo_stations_merged - describe.loc['mean']) / describe.loc['std']
    df_norm.to_csv("test/%s_meo_norm_data.csv" %(city))


