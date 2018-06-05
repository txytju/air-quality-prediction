# coding: utf-8
import xgboost
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from model.xgboost.xgboost_data_util import generate_training_set, generate_dev_set, generate_X_test_set
from metrics.metrics import smape
from model.model_data_util import day_mean, day_range, expand_day_trend
from model.model_data_util import get_stations_filters, get_X_feature_filters, get_y_feature_filters
from model.day_trend import daily_statistics

from utils.information import bj_station_list, bj_X_aq_list, bj_y_aq_list
from utils.information import ld_station_list, ld_X_aq_list, ld_y_aq_list, ld_X_meo_list
bj_X_meo_list = ["temperature","pressure","humidity","direction","speed"]



def train_xgboost(city, type_of_feature):


    if city =="bj":
        station_list = bj_station_list
        X_aq_list = bj_X_aq_list
        y_aq_list = bj_y_aq_list
        X_meo_list = bj_X_meo_list
    elif city == "ld":
        station_list = ld_station_list
        X_aq_list = ld_X_aq_list
        y_aq_list = ld_y_aq_list
        X_meo_list = ld_X_meo_list  


    day_trend = daily_statistics(city=city,
                                 station_list=station_list,
                                 X_aq_list=X_aq_list,
                                 y_aq_list=y_aq_list,
                                 X_meo_list=X_meo_list,)  # shape of (24, features)


    X_train_mean, y_train_mean = generate_training_set(city="bj", 
                                                      station_list=station_list, 
                                                      X_aq_list=X_aq_list, 
                                                      y_aq_list=y_aq_list, 
                                                      X_meo_list=X_meo_list, 
                                                      use_day=True, 
                                                      pre_days=5, 
                                                      batch_size=32, 
                                                      gap=1,
                                                      use_day_model=True,
                                                      generate_mean=True)

    X_train_range, y_train_range = generate_training_set(city="bj", 
                                                      station_list=station_list, 
                                                      X_aq_list=X_aq_list, 
                                                      y_aq_list=y_aq_list, 
                                                      X_meo_list=X_meo_list, 
                                                      use_day=True, 
                                                      pre_days=5, 
                                                      batch_size=32, 
                                                      gap=1,
                                                      use_day_model=True,
                                                      generate_range=True)

    X_dev_mean, y_dev_mean, y_dev_original = generate_dev_set(city="bj", 
                                                      station_list=station_list, 
                                                      X_aq_list=X_aq_list, 
                                                      y_aq_list=y_aq_list, 
                                                      X_meo_list=X_meo_list,  
                                                      pre_days=5, 
                                                      gap=1,
                                                      use_day_model=True,
                                                      generate_mean=True)

    X_dev_range, y_dev_range, y_dev_original = generate_dev_set(city="bj", 
                                                      station_list=station_list, 
                                                      X_aq_list=X_aq_list, 
                                                      y_aq_list=y_aq_list, 
                                                      X_meo_list=X_meo_list,  
                                                      pre_days=5, 
                                                      gap=1,
                                                      use_day_model=True,
                                                      generate_range=True)

    X_test_mean =  generate_X_test_set(city="bj", 
                            station_list=station_list, 
                            X_aq_list=X_aq_list, 
                            X_meo_list=X_meo_list, 
                            pre_days=5, 
                            gap=1,
                            use_day_model=True,
                            generate_mean=True,
                            generate_range=False)



    X_test_range =  generate_X_test_set(city="bj", 
                            station_list=station_list, 
                            X_aq_list=X_aq_list, 
                            X_meo_list=X_meo_list, 
                            pre_days=5, 
                            gap=1,
                            use_day_model=True,
                            generate_range=True)


    print(X_train_mean.shape, y_train_mean.shape)
    print(X_dev_mean.shape, y_dev_mean.shape, y_dev_original.shape)


    if type_of_feature == "mean" :

        X_train = X_train_mean
        y_train = y_train_mean
        X_dev = X_dev_mean
        y_dev = y_dev_mean
        X_test = X_test_mean

    elif type_of_feature == "range" :

        X_train = X_train_range
        y_train = y_train_range
        X_dev = X_dev_range
        y_dev = y_dev_range
        X_test = X_test_range

    # train the model
    y_dev_pred = np.zeros(y_dev.shape)
    y_test_pred = np.zeros((1,210))

    for i in range(y_train.shape[1]):
        model = XGBRegressor(n_estimators=300, max_depth=5)
        model.fit(X_train, y_train[:,i])
        p_dev = model.predict(X_dev)    
        p_test = model.predict(X_test)
        print(i)
        y_dev_pred[:,i] = p_dev
        y_test_pred[:,i] = p_test


    delta = y_dev - y_dev_pred
    error = np.mean(np.abs(delta))
    print("%s error : %f" %(type_of_feature, error))

    # if training on mean, use average day_trend to predict and caculate smape on dev set
    if type_of_feature == "mean" : 
        y_dev_pred_reshape = y_dev_pred.reshape((28,2,105))
        y1 = y_dev_pred_reshape[:,0,:]
        y2 = y_dev_pred_reshape[:,1,:]
        y1 = np.expand_dims(y1, 1)
        y2 = np.expand_dims(y2, 1)
        y1 = np.repeat(y1, 24, axis=1)
        y2 = np.repeat(y2, 24, axis=1)
        y_dev_pred_full = np.concatenate([y1, y2], axis = 1)  # (28, 48, 105)

        m = y_dev_pred_full.shape[0]
        unscaled_day_trend = expand_day_trend(m, day_trend)

        y_dev_pred_full += unscaled_day_trend

        smape_dev = smape(y_dev_original, y_dev_pred_full)
        print("smape is : %f" %(smape_dev))

        return smape_dev

    return None