
## 主程序
import numpy as np
import pandas as pd


def write_value_to_csv(city,
                       file_name, 
                       values, 
                       output_features, 
                       day=True, 
                       seperate=False,
                       one_day_model=False):
    '''
    write all the values to a csv file according to output_features.

    day : False to use old model, True to use day model.
    '''
    # just use the last element of values
    # values shape is (m, output_hours, output_features)
    # values = values[-1,:,:]

    df_list = []

    for i in range(values.shape[0]) :

        value = values[i,:,:]

        # load sample
        if city == "bj" : 
            df = pd.read_csv("submission/sample_bj_submission.csv")
            features = ["PM2.5", "PM10", "O3"]
        elif city =="ld" :
            df = pd.read_csv("submission/sample_ld_submission.csv")
            features = ["PM2.5", "PM10"]

        df["PM2.5"] = df["PM2.5"].astype('float64')
        df["PM10"] = df["PM10"].astype('float64')

        for index in df.index:
            test_id = df.test_id[index]
            station, hour = test_id.split("#")
            
            for feature in features:
                r = get_value_from_array(value, output_features, station, int(hour), feature)
                # print(r)      
                df.set_value(index, feature, r)

        df.set_index("test_id", inplace=True)
        df[df<0]=0

        # rename columns
        original_names = df.columns.values.tolist()
        names_dict = {original_name : original_name + "_" + str(i) for original_name in original_names}
        df.rename(index=str, columns=names_dict, inplace=True)

        df_list.append(df)

    df = pd.concat(df_list, axis=1)

    if seperate :
        df.to_csv("model_preds_seperate/%s/%s.csv" %(city, file_name))        
    if day :
        df.to_csv("model_preds_day/%s/%s.csv" %(city, file_name))
    if one_day_model :
        df.to_csv("model_preds_one_day/%s/%s.csv" %(city, file_name))       
    else :
        df.to_csv("model_preds/%s/%s.csv" %(city, file_name))


def get_value_from_array(value_array, output_features, target_station, target_hour, target_feature):

    for index, output_feature in enumerate(output_features) :
        features = output_feature.split("_")
        if "aq" in features :
            features.remove("aq")
        station, feature = features
        if "aq" in target_station : 
            target_station = target_station.split("_")[0]
        if target_station in station and feature == target_feature :

            return value_array[target_hour,index]
    return -1
