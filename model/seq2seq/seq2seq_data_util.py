import numpy as np
import pandas as pd

from model.model_data_util import get_training_statistics, day_mean, day_range
from model.model_data_util import get_stations_filters, get_X_feature_filters, get_y_feature_filters


def generate_meo_forecast(city="bj", 
                          station_list=None, 
                          X_meo_list=None, 
                          use_norm_data=True):

    if use_norm_data :
        meo_pred = pd.read_csv("preprocessed_data/after_split/norm_data/%s_pred_meo_norm_data.csv" %(city))    
    
    station_filters = get_stations_filters(meo_pred, station_list)
    meo_feature_filters = get_X_feature_filters(None, X_meo_list, station_filters)

    meo_preds = meo_pred[meo_feature_filters]
    meo_preds = np.expand_dims(meo_preds, axis=0)
    
    return meo_preds


def generate_training_list(city="bj", 
                          station_list=None, 
                          X_aq_list=None, 
                          y_aq_list=None, 
                          X_meo_list=None, 
                          use_day=True, 
                          pre_days=5, 
                          num_training_examples=2000, 
                          gap=0,
                          use_norm_data=True,
                          predict_one_day=False):
    '''
    Generate a large training data list.
    
    Args:
        station_list : a list of used stations.
        X_aq_list : a list of used aq features as input.
        y_aq_list : a list of used aq features as output. 
        X_meo_list : a list of used meo features.
        use_day : bool, True to just use 0-24 h days.
        pre_days : use pre_days history days to predict.
        batch_size : batch_size.
        gap : 0 or 12 or 24. 
                0 : 当天 23点以后进行的模型训练
                12 : 当天中午进行的模型训练
                24 : 不使用当天数据进行的训练
    '''
    if use_norm_data :
        aq_train = pd.read_csv("preprocessed_data/after_split/norm_data/%s_aq_train_data.csv" %(city))
        meo_train = pd.read_csv("preprocessed_data/after_split/norm_data/%s_meo_train_data.csv" %(city))
    else :
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

    if predict_one_day :
        y_hours =  23
    else :
        y_hours =  47
    
    for i in range(num_training_examples):       
        flag = True        
        while flag :

            X_start_index = int(np.random.choice(total_start_points, 1, replace = False))
            X_end_index = X_start_index + pre_days * 24 - 1 - gap

            y_start_index =  X_start_index + pre_days * 24
            y_end_index = y_start_index + y_hours


            # print(X_start_index, X_end_index, y_start_index, y_end_index)

            X = X_df.loc[X_start_index : X_end_index]
            y = y_df.loc[y_start_index : y_end_index]

            # 判断是不是有 NAN
            if pd.isnull(X).any().any() or pd.isnull(y).any().any():
                pass
            else :     
                X_df_list.append(X)
                y_df_list.append(y)
                flag = False


    return X_df_list, y_df_list


def oversampling_training_list(X_df_list, 
                               y_df_list, 
                               oversample_part=0.2, 
                               repeats=3):

    # get_all_pm2.5 column names
    y_df_example = y_df_list[0]
    pm25_columns = []
    for column in y_df_example.columns :
        if "PM2.5" in column :
            pm25_columns.append(column)  # a list of column names 
    
    pm25_mean_list = []
    for y_df in y_df_list :
        pm25_mean = y_df[pm25_columns].mean().mean()
        pm25_mean_list.append(pm25_mean)

    # sort X_df_list and y_df_list according to pm25_mean_list
    X_df_list = [X for _,X in zip(pm25_mean_list, X_df_list)]   # from small to big
    y_df_list = [y for _,y in zip(pm25_mean_list, y_df_list)]

    # oversmapling
    oversample_num = int(oversample_part * len(X_df_list))
    X_df_list = X_df_list + repeats * X_df_list[:oversample_num]
    y_df_list = y_df_list + repeats * y_df_list[:oversample_num]   
     
    X_array_list = []
    y_array_list = []

    for X in X_df_list :
        X = np.array(X)
        X = np.expand_dims(X, axis=0)
        X_array_list.append(X)

    for y in y_df_list :
        y = np.array(y)
        y = np.expand_dims(y, axis=0)
        y_array_list.append(y)

    return X_array_list, y_array_list


def generate_training_batch(X_array_list, 
                            y_array_list,
                            batch_size=128,
                            use_day_model=True,
                            generate_mean=False,
                            generate_range=False):
    
    X_list = []
    y_list = []

    for i in range(batch_size):       
            selected_index = int(np.random.choice(len(X_array_list), 1, replace = False))
            X_list.append(X_array_list[selected_index])
            y_list.append(y_array_list[selected_index])

    X_train_batch = np.concatenate(X_list, axis=0)
    y_train_batch = np.concatenate(y_list, axis=0)
    
    if use_day_model :

        if generate_mean and generate_range :
            X_train_batch_mean = day_mean(X_train_batch)
            y_train_batch_mean = day_mean(y_train_batch)
            X_train_batch_range = day_range(X_train_batch)
            y_train_batch_range = day_range(y_train_batch)            
            # concanate 
            X_train_batch_all = np.concatenate([X_train_batch_mean, X_train_batch_range], axis=2)
            y_train_batch_all = np.concatenate([y_train_batch_mean, y_train_batch_range], axis=2)
            
            return X_train_batch_all, (y_train_batch_all, y_train_batch)
        
        elif generate_mean and not generate_range : 
            X_train_batch_mean = day_mean(X_train_batch)
            y_train_batch_mean = day_mean(y_train_batch)
            
            return X_train_batch_mean, (y_train_batch_mean, y_train_batch)
        
        elif generate_range and not generate_mean :
            X_train_batch_range = day_range(X_train_batch)
            y_train_batch_range = day_range(y_train_batch)
            
            return X_train_batch_range, (y_train_batch_range, y_train_batch)
        
        else :
            print("wrong mode !")
    
    else : 
        
        return X_train_batch, y_train_batch


def generate_X_test_set(city="bj", 
                        station_list=None, 
                        X_aq_list=None, 
                        X_meo_list=None, 
                        pre_days=5, 
                        gap=0,
                        use_norm_data=True,
                        use_day_model=True,
                        generate_mean=False,
                        generate_range=False) :

    if use_norm_data :
        aq_dev = pd.read_csv("preprocessed_data/after_split/norm_data/%s_aq_dev_data.csv" %(city))
        meo_dev = pd.read_csv("preprocessed_data/after_split/norm_data/%s_meo_dev_data.csv" %(city))
    else :
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
        if generate_mean and generate_range :
            X_mean =  day_mean(X)
            X_range = day_range(X)
            X = np.concatenate([X_mean, X_range], axis=2)
        elif generate_mean and not generate_range : 
            X = day_mean(X)
        elif generate_range and not generate_mean :
            X = day_range(X)

    return X


def generate_aggr_set(city="bj", 
                      station_list=None, 
                      X_aq_list=None, 
                      y_aq_list=None, 
                      X_meo_list=None, 
                      pre_days=5, 
                      gap=0,
                      use_day_model=True,
                      use_norm_data=True,
                      generate_mean=False,
                      generate_range=False,
                      aggr_start_time="2018-5-22 0:00",
                      predict_one_day=False):
    '''   
    Args:
        station_list : a list of used stations.
        X_aq_list : a list of used aq features as input.
        y_aq_list : a list of used aq features as output. 
        X_meo_list : a list of used meo features.
        aggr_start_time : set aggr y start time.
    '''
    if use_norm_data :
        aq_aggr = pd.read_csv("preprocessed_data/after_split/norm_data/%s_aq_dev_data.csv" %(city))
        meo_aggr = pd.read_csv("preprocessed_data/after_split/norm_data/%s_meo_dev_data.csv" %(city))
    else :
        aq_aggr = pd.read_csv("preprocessed_data/after_split/original_data/%s_aq_dev_data.csv" %(city))
        meo_aggr = pd.read_csv("preprocessed_data/after_split/original_data/%s_meo_dev_data.csv" %(city))

    aggr_df = pd.concat([aq_aggr, meo_aggr], axis=1)

    # remove dupilicated columns
    aggr_df = aggr_df.loc[:,~aggr_df.columns.duplicated()]
    aggr_df['time'] = pd.to_datetime(aggr_df['time'])
    
    # get the index of the aggr_start_time
    total_y_start_index = aggr_df.loc[aggr_df['time'] == aggr_start_time].index.values[0]

    station_filters = get_stations_filters(aggr_df, station_list)
    X_feature_filters = get_X_feature_filters(X_aq_list, X_meo_list, station_filters)
    y_feature_filters = get_y_feature_filters(y_aq_list, station_filters)

    X_df = aggr_df[X_feature_filters]
    y_df = aggr_df[y_feature_filters]   


    # step 2 : generate aggr batch
    X_df_list = []
    y_df_list = []

    if predict_one_day :
        y_day_one_df_list = []
    
    if predict_one_day :
        y_hours = 23
    else :
        y_hours = 47

    for y_start_index in range(total_y_start_index, y_df.shape[0]-y_hours, 24):
        
        y_end_index = y_start_index + y_hours
        X_start_index = y_start_index - pre_days * 24
        X_end_index = y_start_index - 1 - gap

        y = y_df.loc[y_start_index : y_end_index]
        X = X_df.loc[X_start_index : X_end_index]

        X = np.array(X)
        y = np.array(y)
        X = np.expand_dims(X, axis=0)
        y = np.expand_dims(y, axis=0)
        X_df_list.append(X)
        y_df_list.append(y)

        if predict_one_day : 
            y_day_one = X_df.loc[y_start_index : y_end_index - 24]  # just for the 1 st day
            y_day_one = np.array(y_day_one)
            y_day_one = np.expand_dims(y_day_one, axis=0) 
            y_day_one_df_list.append(y_day_one)



    X_aggr = np.concatenate(X_df_list, axis=0)
    y_aggr = np.concatenate(y_df_list, axis=0)

    if predict_one_day :
        x_aggr_day_one = np.concatenate(y_day_one_df_list, axis=0)
    

    if use_day_model :

        if generate_mean and generate_range :
            X_aggr_mean = day_mean(X_aggr)
            y_aggr_mean = day_mean(y_aggr)
            X_aggr_range = day_range(X_aggr)
            y_aggr_range = day_range(y_aggr)            
            # concanate 
            X_aggr_all = np.concatenate([X_aggr_mean, X_aggr_range], axis=2)
            y_aggr_all = np.concatenate([y_aggr_mean, y_aggr_range], axis=2)
            
            return X_aggr_all, (y_aggr_all, y_aggr)
        
        elif generate_mean and not generate_range : 
            X_aggr_mean = day_mean(X_aggr)
            y_aggr_mean = day_mean(y_aggr)
            
            return X_aggr_mean, (y_aggr_mean, y_aggr)
        
        elif generate_range and not generate_mean :
            X_aggr_range = day_range(X_aggr)
            y_aggr_range = day_range(y_aggr)
            
            return X_aggr_range, (y_aggr_range, y_aggr)
        
        else :
            print("wrong mode !")
    
    elif predict_one_day : 

        return X_aggr, y_aggr, x_aggr_day_one

    else :
        return X_aggr, y_aggr



def generate_dev_set(city="bj", 
                     station_list=None, 
                     X_aq_list=None, 
                     y_aq_list=None, 
                     X_meo_list=None, 
                     pre_days=5, 
                     gap=0,
                     use_day_model=True,
                     use_norm_data=True,
                     generate_mean=False,
                     generate_range=False,
                     for_aggr=False,
                     aggr_start_time="2018-5-22 0:00",
                     predict_one_day=False):
    '''
   
    Args:
        station_list : a list of used stations.
        X_aq_list : a list of used aq features as input.
        y_aq_list : a list of used aq features as output. 
        X_meo_list : a list of used meo features.
        use_day_model : which model to use, True for day_based model, false for old model.
    '''
    if use_norm_data :
        aq_dev = pd.read_csv("preprocessed_data/after_split/norm_data/%s_aq_dev_data.csv" %(city))
        meo_dev = pd.read_csv("preprocessed_data/after_split/norm_data/%s_meo_dev_data.csv" %(city))
    else :
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

    if predict_one_day :
        y_hours = 23
    else :
        y_hours = 47

    if for_aggr :
        # remove dupilicated columns
        dev_df = dev_df.loc[:,~dev_df.columns.duplicated()]
        dev_df['time'] = pd.to_datetime(dev_df['time'])
        # get the index of the aggr_start_time
        max_y_start_index = dev_df.loc[dev_df['time'] == aggr_start_time].index.values[0] - y_hours

    else :
        max_y_start_index = X_df.shape[0] - 2 * 24  # if not for aggr, use all data from the dev set

    X_df_list = []
    y_df_list = []

    if predict_one_day :
        y_day_one_df_list = []


    for y_start_index in range(min_y_start_index, max_y_start_index, 24):

        X_start_index = y_start_index - pre_days * 24
        X_end_index = y_start_index - 1 - gap

        y_end_index = y_start_index + y_hours


        X = X_df.loc[X_start_index : X_end_index]
        y = y_df.loc[y_start_index : y_end_index]

        X = np.array(X)
        y = np.array(y)
        X = np.expand_dims(X, axis=0)
        y = np.expand_dims(y, axis=0)
        X_df_list.append(X)
        y_df_list.append(y)

        if predict_one_day :
            y_day_one = X_df.loc[y_start_index : y_end_index - 24]  # just for the 1 st day
            y_day_one = np.array(y_day_one)
            y_day_one = np.expand_dims(y_day_one, axis=0) 
            y_day_one_df_list.append(y_day_one)


    X_dev = np.concatenate(X_df_list, axis=0)
    y_dev = np.concatenate(y_df_list, axis=0)

    if predict_one_day :
        x_dev_day_one = np.concatenate(y_day_one_df_list, axis=0)
    
    if use_day_model :

        if generate_mean and generate_range :
            X_dev_mean = day_mean(X_dev)
            y_dev_mean = day_mean(y_dev)
            X_dev_range = day_range(X_dev)
            y_dev_range = day_range(y_dev)            
            # concanate 
            X_dev_all = np.concatenate([X_dev_mean, X_dev_range], axis=2)
            y_dev_all = np.concatenate([y_dev_mean, y_dev_range], axis=2)
            
            return X_dev_all, (y_dev_all, y_dev)     # (m, days_in, 2*input_features), (m, days_out, 2*output_features)
        
        elif generate_mean and not generate_range : 
            X_dev_mean = day_mean(X_dev)
            y_dev_mean = day_mean(y_dev)
            
            return X_dev_mean, (y_dev_mean, y_dev)
        
        elif generate_range and not generate_mean :
            X_dev_range = day_range(X_dev)
            y_dev_range = day_range(y_dev)
            
            return X_dev_range, (y_dev_range, y_dev)
        
        else :
            print("wrong mode !")
    
    elif predict_one_day : 
        
        return X_dev, y_dev, x_dev_day_one, X_feature_filters, y_feature_filters
    
    else :
        
        return X_dev, y_dev



def pad(y_day_one, dev_x_day_one, X_feature_filters, y_feature_filters):
    '''
    Transfor model predict shape of (m,length,105) to (m,length,385)
    X_feature_filters include y_feature_filters
    '''
    result = np.zeros(dev_x_day_one.shape)

    for i in range(len(X_feature_filters)):
        feature = X_feature_filters[i]
        if feature in y_feature_filters :
            position = y_feature_filters.index(feature)
            result[:,:,i] = y_day_one[:,:,position]
        else :
            result[:,:,i] = dev_x_day_one[:,:,i]

    return result


def concate_aq_meo(aq_day_one, meo_pred_day_one, X_feature_filters, y_aq_list, X_meo_list):
    '''
    Transfor model predict shape of (m,length,105) to (m,length,385) using weature prediction
    X_feature_filters is in order, y_aq_list and X_meo_list are not.
    '''
    y_aq_list.sort()
    X_meo_list.sort()

    (m, length, aq_features) = aq_day_one.shape
    (_, _, meo_features) = meo_pred_day_one.shape

    print(aq_features, meo_features)

    assert aq_features + meo_features == len(X_feature_filters)

    result = np.zeros((m, length, len(X_feature_filters)))

    for i in range(len(X_feature_filters)):
        feature =X_feature_filters[i]
        
        if feature in y_aq_list :
            position = y_aq_list.index(feature)
            result[:,:,i] = aq_day_one[:,:,position]
        

        elif feature in X_meo_list :
            position =X_meo_list.index(feature)            
            result[:,:,i] = meo_pred_day_one[:,:,i]

    print(result.shape)

    return result


    
# # 可以按照 站点名称，特征名称，数据天数来灵活的生成验证集
# def generate_training_set(city="bj", 
#                           station_list=None, 
#                           X_aq_list=None, 
#                           y_aq_list=None, 
#                           X_meo_list=None, 
#                           use_day=True, 
#                           pre_days=5, 
#                           batch_size=32, 
#                           gap=0,
#                           use_norm_data=True,
#                           use_day_model=True,
#                           generate_mean=False,
#                           generate_range=False,
#                           predict_one_day=False):
#     '''
    
#     Args:
#         station_list : a list of used stations.
#         X_aq_list : a list of used aq features as input.
#         y_aq_list : a list of used aq features as output. 
#         X_meo_list : a list of used meo features.
#         use_day : bool, True to just use 0-24 h days.
#         pre_days : use pre_days history days to predict.
#         batch_size : batch_size.
#         gap : 0 or 12 or 24. 
#                 0 : 当天 23点以后进行的模型训练
#                 12 : 当天中午进行的模型训练
#                 24 : 不使用当天数据进行的训练
#     '''
#     if use_norm_data :
#         aq_train = pd.read_csv("preprocessed_data/after_split/norm_data/%s_aq_train_data.csv" %(city))
#         meo_train = pd.read_csv("preprocessed_data/after_split/norm_data/%s_meo_train_data.csv" %(city))
#     else :
#         aq_train = pd.read_csv("preprocessed_data/after_split/original_data/%s_aq_train_data.csv" %(city))
#         meo_train = pd.read_csv("preprocessed_data/after_split/original_data/%s_meo_train_data.csv" %(city))
        
#     train_df = pd.concat([aq_train, meo_train], axis=1)
    
#     station_filters = get_stations_filters(train_df, station_list)
#     X_feature_filters = get_X_feature_filters(X_aq_list, X_meo_list, station_filters)
#     y_feature_filters = get_y_feature_filters(y_aq_list, station_filters)
    
#     X_df = train_df[X_feature_filters]
#     y_df = train_df[y_feature_filters]

    

#     # step 4 : generate training batch
#     X_df_list = []
#     y_df_list = []
    
#     max_start_points = X_df.shape[0] - (pre_days + 2) * 24 - gap
#     if use_day : 
#         total_start_points = range(0, max_start_points, 24)
#     else :
#         total_start_points = range(0, max_start_points, 1)

#     if predict_one_day :
#         y_hours =  23
#     else :
#         y_hours =  47
    
#     for i in range(batch_size):       
#         flag = True        
#         while flag :

#             X_start_index = int(np.random.choice(total_start_points, 1, replace = False))
#             X_end_index = X_start_index + pre_days * 24 - 1 - gap

#             y_start_index =  X_start_index + pre_days * 24
#             y_end_index = y_start_index + y_hours


#             # print(X_start_index, X_end_index, y_start_index, y_end_index)

#             X = X_df.loc[X_start_index : X_end_index]
#             y = y_df.loc[y_start_index : y_end_index]

#             # 判断是不是有 NAN
#             if pd.isnull(X).any().any() or pd.isnull(y).any().any():
#                 pass
#             else :     
#                 X = np.array(X)
#                 y = np.array(y)
#                 X = np.expand_dims(X, axis=0)
#                 y = np.expand_dims(y, axis=0)
#                 X_df_list.append(X)
#                 y_df_list.append(y)
#                 flag = False

#     X_train_batch = np.concatenate(X_df_list, axis=0)
#     y_train_batch = np.concatenate(y_df_list, axis=0)
    
#     if use_day_model :

#         if generate_mean and generate_range :
#             X_train_batch_mean = day_mean(X_train_batch)
#             y_train_batch_mean = day_mean(y_train_batch)
#             X_train_batch_range = day_range(X_train_batch)
#             y_train_batch_range = day_range(y_train_batch)            
#             # concanate 
#             X_train_batch_all = np.concatenate([X_train_batch_mean, X_train_batch_range], axis=2)
#             y_train_batch_all = np.concatenate([y_train_batch_mean, y_train_batch_range], axis=2)
            
#             return X_train_batch_all, (y_train_batch_all, y_train_batch)
        
#         elif generate_mean and not generate_range : 
#             X_train_batch_mean = day_mean(X_train_batch)
#             y_train_batch_mean = day_mean(y_train_batch)
            
#             return X_train_batch_mean, (y_train_batch_mean, y_train_batch)
        
#         elif generate_range and not generate_mean :
#             X_train_batch_range = day_range(X_train_batch)
#             y_train_batch_range = day_range(y_train_batch)
            
#             return X_train_batch_range, (y_train_batch_range, y_train_batch)
        
#         else :
#             print("wrong mode !")
    
#     else : 
        
#         return X_train_batch, y_train_batch
