import pandas as pd
import numpy as np

def daily_statistics(city="bj",
                     station_list=None,
                     X_aq_list=None,
                     y_aq_list=None,
                     X_meo_list=None):
    
    '''
    daily change compared with mean value of features.
    '''
    output_features = []
    for station in station_list : 
        for aq_feature in y_aq_list :
            output_features.append(station + "_" + aq_feature)
    output_features.sort()

    aq_train = pd.read_csv("preprocessed_data/after_split/norm_data/%s_aq_train_data.csv" %(city))
    
    if city =="bj" :
        aq_train = aq_train.loc[:10871]
    elif city == "ld" :
        aq_train = aq_train.loc[:10871]

    aq_train = aq_train[output_features]


    l = []

    for i in range(0,len(aq_train),24):

        df = aq_train.loc[i:i+23]
        # print(df.shape)
        df = df.reset_index()
        df.drop(columns=["index"], inplace=True)
        df = df - df.mean()
        
        if not df.isnull().values.any() : 
            l.append(df)


    y = pd.DataFrame(np.zeros(l[0].shape), columns=l[0].columns)
    for i in l:
        y += i
    y = y / len(l)

    return y  # shape of (24, features)


def get_training_statistics(city):
    '''
    Get statics values of aq and meo data.
    '''
    aq_describe = pd.read_csv("preprocessed_data/before_split/%s_aq_describe.csv" %(city))
    aq_describe.set_index("Unnamed: 0", inplace=True)
    
    meo_describe = pd.read_csv("preprocessed_data/before_split/%s_meo_describe.csv" %(city))
    meo_describe.set_index("Unnamed: 0", inplace=True)  
    
    statistics = pd.concat([aq_describe, meo_describe], axis=1)
    statistics = statistics.loc[["mean", "std"]]
    return statistics


def day_mean(data_batch):
    '''
    data_batch是按照小时计的数据，计算data_batch转换成按天的统计量
    data_batch : shape of (m, hours, features)
    '''
    days = []

    for i in range(0,data_batch.shape[1],24):
        X = data_batch[:,i:i+24,:]
        X = X.mean(axis=1, keepdims=True)
        days.append(X)

    days = np.concatenate(days, axis=1)

    return days

def day_range(data_batch):
    '''
    data_batch是按照小时计的数据，计算data_batch转换成按天的统计量
    data_batch : shape of (m, hours, features)
    '''
    days = []
    
    for i in range(0,data_batch.shape[1],24):
        X = data_batch[:,i:i+24,:]
        X_max = X.max(axis=1, keepdims=True)
        X_min = X.min(axis=1, keepdims=True)
        X_range = X_max - X_min
        days.append(X_range)

    days = np.concatenate(days, axis=1)

    return days


def get_stations_filters(df, station_list):
    station_filters = []
    for station in station_list : 
        station_filter = [index for index in df.columns if station in index]
        station_filters += station_filter

    return station_filters


def get_X_feature_filters(X_aq_list, X_meo_list, station_filters):
    # step 2 : filter of X features
    X_feature_filters = []
    
    if X_meo_list :
        if X_aq_list :
            X_features = X_aq_list + X_meo_list
        else :
            X_features = X_meo_list
    else :
        X_features = X_aq_list
        
    for i in station_filters : 
        if i.split("_")[-1] in X_features :
            X_feature_filters += [i]
            
    X_feature_filters.sort()  # 排序，保证训练集和验证集中的特征的顺序一致
    
    return X_feature_filters
  

def get_y_feature_filters(y_aq_list, station_filters):
    y_feature_filters = []
    y_features = y_aq_list
    
    for i in station_filters : 
        if i.split("_")[-1] in y_features :
            y_feature_filters += [i]
    
    y_feature_filters.sort()  # 排序，保证训练集和验证集中的特征的顺序一致

    return y_feature_filters


def expand_final_preds_dim(final_preds):

    # convert day_mean to hour based on daily trend
    final_preds_1 = final_preds[:,0,:]  # (m, features)
    final_preds_2 = final_preds[:,1,:]  # (m, features)

    final_preds_1 = final_preds_1[:,None,:]  # (m, 1, features)
    final_preds_2 = final_preds_2[:,None,:]  # (m, 1, features)


    final_preds_1 = np.repeat(final_preds_1, 24, axis=1)  # (m, 24, features)
    final_preds_2 = np.repeat(final_preds_2, 24, axis=1)  # (m, 24, features)

    final_preds = np.concatenate([final_preds_1, final_preds_2], axis = 1)  # (m, 48, features)

    return final_preds


def expand_day_trend(m, day_trend):
    '''
    m : data barch size.
    day_trend : day_trend to expand.
    '''
    day_trend = np.expand_dims(day_trend, axis=0)  # (24, features) -> (1, 24, features)
    day_trend = np.tile(day_trend, (m,2,1))  # (1, 24, features) -> (m, 48, features)

    return day_trend



def scale_day_trend(target_range, unscaled_day_trend):
    '''
    target_range : part of final_preds, shape of (m, 2, features)
    day_trend :  day trend after "expand_day_trend" function ,shape of (m, 48, features)
    '''
    # print(target_range.shape)
    # print(unscaled_day_trend.shape)

    day_one_trend = unscaled_day_trend[:, :24, :]  # (m, 24, features)
    day_two_trend = unscaled_day_trend[:, 24:, :]  # (m, 24, features)

    max_1 = day_one_trend.max(axis=1)     # (m, features)
    min_1 = day_one_trend.min(axis=1)     # (m, features)
    range_1 = max_1 - min_1             # (m, features)
    # print(range_1.shape)
    # print(range_1)

    max_2 = day_two_trend.max(axis=1)     # (m, features)
    min_2 = day_two_trend.min(axis=1)     # (m, features)
    range_2 = max_2 - min_2             # (m, features)

    traget_range_1 = target_range[:,0,:]  # (m, features)
    traget_range_2 = target_range[:,1,:]  # (m, features)

    scale_1 = np.divide(traget_range_1, range_1)  # (m, features)
    scale_2 = np.divide(traget_range_2, range_2)  # (m, features)
    
    # print(scale_1)
    
    scale_1 = np.expand_dims(scale_1, 1)  # (m, 1, features)
    scale_2 = np.expand_dims(scale_2, 1)  # (m, 1, features)


    scale_1 = np.repeat(scale_1, 24, axis=1)  # (m, 24, features)
    scale_2 = np.repeat(scale_2, 24, axis=1)  # (m, 24, features)

    scale = np.concatenate([scale_1, scale_2], axis = 1)  # (m, 48, features)

    day_trend = np.multiply(scale, unscaled_day_trend)

    return day_trend
