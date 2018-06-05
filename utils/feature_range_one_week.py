# coding: utf-8
# 分析空气质量特征的日内变化幅度和哪些特征有关
# 现象：连续两天的某个特征的日内变化的幅度相差很大，分析可能和哪些因素有关？
# 可能的原因：星期几？节假日？当天/之前一天的天气？

# 星期几

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from utils.information import bj_station_list, bj_X_aq_list, bj_y_aq_list
from utils.information import ld_station_list, ld_X_aq_list, ld_y_aq_list, ld_X_meo_list
bj_X_meo_list = ["temperature","pressure","humidity","direction","speed"]


def station_feature_week(city, df, station_class, aq_feature, delta_list):
    '''
    Args :
        delta_list : a list of pd Series, everyone of them is a day range.
        station_class : a list of station names which share something in common.
        aq_feature : aq feature that you concern, can be "PM2.5", "PM10" or "O3"
    Return : 
        a dict with column name as key and range of average_feature_week as value.
    '''
    dic = {}
    for column in df.columns :
        if city=="bj":
            station, aq, feature = column.split("_")
            station = station + "_" + aq
        elif city=="ld":
            station, feature = column.split("_")
        if station in station_class and feature == aq_feature :
            # print(station)
            features = []

            for delta in delta_list:
                features.append(delta[column])

            feature_weeks = []
            for i in range(0,len(features),7):
                feature_one_week = features[i:i+7]
                if len(feature_one_week) == 7:
                    feature_weeks.append(feature_one_week)

            feature_weeks = np.array(feature_weeks)
            average_feature_week = np.mean(feature_weeks, axis=0)
            dic[column] = average_feature_week
            plt.plot(average_feature_week)
    
    return dic



def get_feature_range_one_week(city="bj"):
    '''
    Return :
        feature_range_week : "range in a day" trend in a week for every feature, shape of (output_features, 7).
        this result is used to scale the 24 hours trend in the predictions.
    '''

    if city=="bj":
        station_list = bj_station_list
        X_aq_list = bj_X_aq_list
        y_aq_list = bj_y_aq_list       
        X_meo_list = bj_X_meo_list

        bj = pd.read_csv("preprocessed_data/before_split/bj_aq_data.csv")
        bj = bj[bj.time > "2017-01-01 23:00:00"]
        bj = bj.reset_index()
        bj.drop(columns=["index"], inplace=True)
        bj.drop(["time"], axis=1, inplace=True)   
        city_df = bj

    elif city=="ld":
        station_list = ld_station_list
        X_aq_list = ld_X_aq_list
        y_aq_list = ld_y_aq_list
        X_meo_list = ld_X_meo_list   

        ld = pd.read_csv("preprocessed_data/before_split/ld_aq_data.csv")
        ld = ld[ld.time > "2017-01-01 23:00:00"]
        ld = ld.reset_index()
        ld.drop(columns=["index"], inplace=True)
        ld.drop(["time"], axis=1, inplace=True) 
        city_df = ld

    output_features = []
    for station in station_list : 
        for aq_feature in y_aq_list :
            output_features.append(station + "_" + aq_feature)
    output_features.sort()

    days = []
    for i in range(0, city_df.shape[0], 24):    
        df = city_df.loc[i:i+23]    
        if not pd.isnull(df).any().any():
            days.append(df)

    delta_list = []
    for i in range(len(days)):
        day_min = days[i].min()  
        day_max = days[i].max()
        delta = day_max - day_min
        delta_list.append(delta)

    if city=="bj" :

        duizhao = ["dingling_aq", "badaling_aq", "miyunshuiku_aq", "donggaocun_aq", "yongledian_aq", "yufa_aq", "liulihe_aq"]
        jiaoqu = ["fangshan_aq", "daxing_aq", "yizhuang_aq", "tongzhou_aq", "shunyi_aq", "pingchang_aq", "mentougou_aq", "pinggu_aq", "huairou_aq", "miyun_aq", "yanqin_aq"]
        jiaotong = ["qianmen_aq", "yongdingmennei_aq", "xizhimenbei_aq", "nansanhuan_aq", "dongsihuan_aq"]
        chengqu = ["dongsi_aq", "tiantan_aq", "guanyuan_aq", "wanshouxigong_aq", "aotizhongxin_aq", "nongzhanguan_aq", "wanliu_aq", "beibuxinqu_aq", "zhiwuyuan_aq", "fengtaihuayuan_aq", "yungang_aq", "gucheng_aq"]


        r_chengqu = station_feature_week(city, city_df, chengqu, "PM2.5", delta_list)
        r_jiaotong = station_feature_week(city, city_df, jiaotong, "PM2.5", delta_list)
        r_duizhao = station_feature_week(city, city_df, duizhao, "PM2.5", delta_list)
        r_jiaoqu = station_feature_week(city, city_df, jiaoqu, "PM2.5", delta_list)
        mean_chengqu_PM25 = np.mean(np.array(list(r_chengqu.values())), axis=0)
        mean_jiaotong_PM25 = np.mean(np.array(list(r_jiaotong.values())), axis=0)
        mean_duizhao_PM25 = np.mean(np.array(list(r_duizhao.values())), axis=0)
        mean_jiaoqu_PM25 = np.mean(np.array(list(r_jiaoqu.values())), axis=0)


        r_chengqu = station_feature_week(city, city_df, chengqu, "PM10", delta_list)
        r_jiaotong = station_feature_week(city, city_df, jiaotong, "PM10", delta_list)
        r_duizhao = station_feature_week(city, city_df, duizhao, "PM10", delta_list)
        r_jiaoqu = station_feature_week(city, city_df, jiaoqu, "PM10", delta_list)
        mean_chengqu_PM10 = np.mean(np.array(list(r_chengqu.values())), axis=0)
        mean_jiaotong_PM10 = np.mean(np.array(list(r_jiaotong.values())), axis=0)
        mean_duizhao_PM10 = np.mean(np.array(list(r_duizhao.values())), axis=0)
        mean_jiaoqu_PM10 = np.mean(np.array(list(r_jiaoqu.values())), axis=0)


        r_chengqu = station_feature_week(city, city_df, chengqu, "O3", delta_list)
        r_jiaotong = station_feature_week(city, city_df, jiaotong, "O3", delta_list)
        r_duizhao = station_feature_week(city, city_df, duizhao, "O3", delta_list)
        r_jiaoqu = station_feature_week(city, city_df, jiaoqu, "O3", delta_list)
        mean_chengqu_O3 = np.mean(np.array(list(r_chengqu.values())), axis=0)
        mean_jiaotong_O3 = np.mean(np.array(list(r_jiaotong.values())), axis=0)
        mean_duizhao_O3 = np.mean(np.array(list(r_duizhao.values())), axis=0)
        mean_jiaoqu_O3 = np.mean(np.array(list(r_jiaoqu.values())), axis=0)

        feature_range_week = np.zeros((len(output_features), 7))

        for i in range(len(output_features)) :
            station, aq, feature = output_features[i].split("_")
            station_name = station + "_" + "aq"
            if station_name in duizhao :
                if feature == "O3":
                    feature_range_week[i,:] = mean_duizhao_O3
                elif feature == "PM10":
                    feature_range_week[i,:] = mean_duizhao_PM10
                elif feature == "PM25":
                    feature_range_week[i,:] = mean_duizhao_PM25
            elif station_name in jiaoqu:
                if feature == "O3":
                    feature_range_week[i,:] = mean_jiaoqu_O3
                elif feature == "PM10":
                    feature_range_week[i,:] = mean_jiaoqu_PM10
                elif feature == "PM25":
                    feature_range_week[i,:] = mean_jiaoqu_PM25
            elif station_name in jiaotong:
                if feature == "O3":
                    feature_range_week[i,:] = mean_jiaotong_O3
                elif feature == "PM10":
                    feature_range_week[i,:] = mean_jiaotong_PM10
                elif feature == "PM25":
                    feature_range_week[i,:] = mean_jiaotong_PM25
            elif station_name in chengqu :
                if feature == "O3":
                    feature_range_week[i,:] = mean_chengqu_O3
                elif feature == "PM10":
                    feature_range_week[i,:] = mean_chengqu_PM10
                elif feature == "PM25":
                    feature_range_week[i,:] = mean_chengqu_PM25

    elif city == "ld" :
        # treat all station in ld the same
        station_classes = ["BL0", "CD1", "CD9", "GN0", "GN3", "GR4", "GR9", "HV1", "KF1", "LW2", "MY7", "ST5", "TH4"]
        r = station_feature_week(city, city_df, station_classes, "PM2.5", delta_list)
        mean_PM25 = np.mean(np.array(list(r.values())), axis=0)
        r = station_feature_week(city, city_df, station_classes, "PM10", delta_list)
        mean_PM10 = np.mean(np.array(list(r.values())), axis=0)

        feature_range_week = np.zeros((len(output_features), 7))


        for i in range(len(output_features)) :
            station_name, feature = output_features[i].split("_")

            if station_name in station_classes :
                if feature == "PM10":
                    feature_range_week[i,:] = mean_PM10
                elif feature == "PM2.5":
                    feature_range_week[i,:] = mean_PM25

    return feature_range_week