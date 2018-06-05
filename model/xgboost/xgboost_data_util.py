import numpy as np
import pandas as pd
from model.model_data_util import get_training_statistics, day_mean, day_range
from model.model_data_util import get_stations_filters, get_X_feature_filters, get_y_feature_filters


def generate_training_set(city="bj", 
                          station_list=None, 
                          X_aq_list=None, 
                          y_aq_list=None, 
                          X_meo_list=None, 
                          use_day=True, 
                          pre_days=5, 
                          gap=0,
                          use_day_model=True,
                          generate_mean=False,
                          generate_range=False):
    '''
    
    Args:
        station_list : a list of used stations.
        X_aq_list : a list of used aq features as input.
        y_aq_list : a list of used aq features as output. 
        X_meo_list : a list of used meo features.
        use_day : bool, True to just use 0-24 h days.
        pre_days : use pre_days history days to predict.
        gap : 0 or 12 or 24. 
                0 : 当天 23点以后进行的模型训练
                12 : 当天中午进行的模型训练
                24 : 不使用当天数据进行的训练
        
    '''
    # aq_train = pd.read_csv("preprocessed_data/after_split/%s_aq_train_data.csv" %(city))
    # meo_train = pd.read_csv("preprocessed_data/after_split/%s_meo_train_data.csv" %(city))
    aq_train = pd.read_csv("preprocessed_data/after_split/original_data/%s_aq_train_data.csv" %(city))
    meo_train = pd.read_csv("preprocessed_data/after_split/original_data/%s_meo_train_data.csv" %(city))   
    
    
    train_df = pd.concat([aq_train, meo_train], axis=1)
    
    station_filters = get_stations_filters(train_df, station_list)
    X_feature_filters = get_X_feature_filters(X_aq_list, X_meo_list, station_filters)
    y_feature_filters = get_y_feature_filters(y_aq_list, station_filters)
    
    X_df = train_df[X_feature_filters]
    y_df = train_df[y_feature_filters]

    

    # step 4 : generate training batch
    X_df_list = []
    y_df_list = []
    
    max_start_points = X_df.shape[0] - (pre_days + 2) * 24 - gap
    if use_day : 
        total_start_points = range(0, max_start_points, 24)
    else :
        total_start_points = range(0, max_start_points, 1)
    

    for X_start_index in total_start_points:       

            X_end_index = X_start_index + pre_days * 24 - 1 - gap

            y_start_index =  X_start_index + pre_days * 24
            y_end_index = y_start_index + 47
            
            # print(X_start_index, X_end_index, y_start_index, y_end_index)

            X = X_df.loc[X_start_index : X_end_index]
            y = y_df.loc[y_start_index : y_end_index]

            # 判断是不是有 NAN
            if pd.isnull(X).any().any() or pd.isnull(y).any().any():
                pass
            else :     
                X = np.array(X)
                y = np.array(y)
                X = np.expand_dims(X, axis=0)
                y = np.expand_dims(y, axis=0)
                X_df_list.append(X)
                y_df_list.append(y)

    X_train = np.concatenate(X_df_list, axis=0)
    y_train = np.concatenate(y_df_list, axis=0)

    if use_day_model :

        if generate_mean :
            X_train_mean = day_mean(X_train) # (m, days, features)
            y_train_mean = day_mean(y_train)
            
            [m,input_day,input_features] = X_train_mean.shape
            [m,output_day,output_features] = y_train_mean.shape
            
            # X_train_mean = X_train_mean.reshape([m, input_day*input_features]) # (m, days*features)
            # y_train_mean = y_train_mean.reshape([m, output_day*output_features])
                  
            return X_train_mean, y_train_mean
        
        elif generate_range :
            X_train_range = day_range(X_train) # (m, days, features)
            y_train_range = day_range(y_train)
            
            [m,input_day,input_features] = X_train_range.shape
            [m,output_day,output_features] = y_train_range.shape
            
            # X_train_range = X_train_range.reshape([m, input_day*input_features]) # (m, days*features)
            # y_train_range = y_train_range.reshape([m, output_day*output_features])
                  
            return X_train_range, y_train_range
    
    return X_train_batch, y_train_batch


def generate_dev_set(city="bj", 
                     station_list=None, 
                     X_aq_list=None, 
                     y_aq_list=None, 
                     X_meo_list=None, 
                     pre_days=5, 
                     gap=0,
                     use_day_model=True,
                     generate_mean=False,
                     generate_range=False):
    '''
   
    Args:
        station_list : a list of used stations.
        X_aq_list : a list of used aq features as input.
        y_aq_list : a list of used aq features as output. 
        X_meo_list : a list of used meo features.
        use_day_model : which model to use, True for day_based model, false for old model.
    '''
    # aq_dev = pd.read_csv("preprocessed_data/after_split/%s_aq_dev_data.csv" %(city))
    # meo_dev = pd.read_csv("preprocessed_data/after_split/%s_meo_dev_data.csv" %(city))
    aq_dev = pd.read_csv("preprocessed_data/after_split/original_data/%s_aq_dev_data.csv" %(city))
    meo_dev = pd.read_csv("preprocessed_data/after_split/original_data/%s_meo_dev_data.csv" %(city))
    
    dev_df = pd.concat([aq_dev, meo_dev], axis=1)
 
    station_filters = get_stations_filters(dev_df, station_list)
    X_feature_filters = get_X_feature_filters(X_aq_list, X_meo_list, station_filters)
    y_feature_filters = get_y_feature_filters(y_aq_list, station_filters)

    X_df = dev_df[X_feature_filters]
    y_df = dev_df[y_feature_filters]   
    
    # step 4 : 按天生成数据
    min_y_start_index = 7 * 24 # 7 是当前的最长的 pre_days
    max_y_start_index = X_df.shape[0] - 2 * 24  # if not for aggr, use all data from the dev set

    X_df_list = []
    y_df_list = []


    for y_start_index in range(min_y_start_index, max_y_start_index, 24):

        X_start_index = y_start_index - pre_days * 24
        X_end_index = y_start_index - 1 - gap

        y_end_index = y_start_index + 47


        X = X_df.loc[X_start_index : X_end_index]
        y = y_df.loc[y_start_index : y_end_index]

        X = np.array(X)
        y = np.array(y)

        X = np.expand_dims(X, axis=0)
        y = np.expand_dims(y, axis=0)

        X_df_list.append(X)
        y_df_list.append(y)

    X_dev = np.concatenate(X_df_list, axis=0)
    y_dev = np.concatenate(y_df_list, axis=0)
    
    if use_day_model :

        if generate_mean :
            X_dev_mean = day_mean(X_dev)
            y_dev_mean = day_mean(y_dev)
            
            [m,input_day,input_features] = X_dev_mean.shape
            [m,output_day,output_features] = y_dev_mean.shape   
            
            X_dev_mean = X_dev_mean.reshape([m, input_day*input_features]) # (m, days*features)
            y_dev_mean = y_dev_mean.reshape([m, output_day*output_features])
            
            return X_dev_mean, y_dev_mean, y_dev   
        
        elif generate_range :
            X_dev_range = day_mean(X_dev)
            y_dev_range = day_mean(y_dev)
            
            [m,input_day,input_features] = X_dev_range.shape
            [m,output_day,output_features] = y_dev_range.shape   
            
            X_dev_range = X_dev_range.reshape([m, input_day*input_features]) # (m, days*features)
            y_dev_range = y_dev_range.reshape([m, output_day*output_features])
            
            return X_dev_range, y_dev_range, y_dev   
    else : 
        
        return X_dev, y_dev


def generate_X_test_set(city="bj", 
                        station_list=None, 
                        X_aq_list=None, 
                        X_meo_list=None, 
                        pre_days=5, 
                        gap=0,
                        use_day_model=True,
                        generate_mean=False,
                        generate_range=False) :


    aq_dev = pd.read_csv("preprocessed_data/after_split/original_data/%s_aq_dev_data.csv" %(city))
    meo_dev = pd.read_csv("preprocessed_data/after_split/original_data/%s_meo_dev_data.csv" %(city))
    
    dev_df = pd.concat([aq_dev, meo_dev], axis=1)

    # step 1 : keep all features about the stations
    station_filters = get_stations_filters(dev_df, station_list)
    
    # step 2 : filter of X features
    X_feature_filters = get_X_feature_filters(X_aq_list, X_meo_list, station_filters)
    X_df = dev_df[X_feature_filters]

    # step 3 : 根据 pre_days 和　gap，确定　preprocessed_data　中Ｘ的值
    delta = 0  

    X_end_index = X_df.shape[0] - 1 - delta

    X_start_index = X_end_index - pre_days * 24 + gap + 1

    X = X_df.loc[X_start_index : X_end_index]
    X = np.array(X)
    X = np.expand_dims(X, axis=0)

    if use_day_model :
        if generate_mean :
            X_mean =  day_mean(X)
            [m,input_day,input_features] = X_mean.shape       
            X_mean = X_mean.reshape([m, input_day*input_features]) # (m, days*features)

            return X_mean
        
        elif generate_range :
            X_range = day_range(X)
            
            [m,input_day,input_features] = X_range.shape     
            X_range = X_range.reshape([m, input_day*input_features]) # (m, days*features)
            
            return X_range
    return X








































