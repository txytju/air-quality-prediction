import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from matplotlib import pyplot as plt
from utils.information import ld_station_list
# %matplotlib inline


def smape(actual, forecast):
    assert len(actual) == len(forecast), "The shape of actual value and forecast value are not the same."
    length = len(actual)
    r = 0
    for i in range(length):
        f = forecast[i]
        a = actual[i]
        r += abs(f-a) / ((abs(a)+abs(f))/2)
        
    return r/length


def caculate_actual_predict_smape(city='bj', 
                                  submit_day = 2,
                                  submit_index = 1):

    # city
    # submit_day = 2 # 提交日是五月的哪一天
    # submit_index = 1 # 当天的第几次提交

    if city == "bj" :
        city_full_name = "Beijing"
    elif city =='ld' : 
        city_full_name = "London"

    # step 1: lead predicted data
    predicted = pd.read_csv("data/submissions/submission_of_cities/%s/my_submission_05%s_%d.csv" %(city, submit_day, submit_index))
    predicted["station_id"] = predicted['test_id'].apply(lambda x : x.split("#")[0])
    predicted.set_index("test_id", inplace=True)
    
    # step 2 : load actual data
    feature = 'aq'
    file_name_1 = "data/%s/%s/%s_%s_2018-05-%d-0_2018-05-%d-23.csv" %(city_full_name, feature, city, feature, submit_day+1, submit_day+1)
    file_name_2 = "data/%s/%s/%s_%s_2018-05-%d-0_2018-05-%d-23.csv" %(city_full_name, feature, city, feature, submit_day+2, submit_day+2)
    df_1 = pd.read_csv(file_name_1)
    df_2 = pd.read_csv(file_name_2)
    actual = pd.concat([df_1, df_2])

    name_pair = {'PM25_Concentration':"PM2.5", 'PM10_Concentration':"PM10",'O3_Concentration':"O3"}
    actual.rename(index=str, columns=name_pair, inplace=True)

    if city == "bj":
        actual = actual[["time","station_id","PM2.5","PM10", "O3"]]
    elif city == "ld":
        actual = actual[["time","station_id","PM2.5","PM10"]]
        
    start_time_str = "2018-05-%d 00:00:00" %(submit_day+1)
    if submit_day+3 < 10 :
        end_time_str = "2018-05-%d 00:00:00" %(submit_day+3)
    else : 
        end_time_str = "2018-05-%d 00:00:00" %(submit_day+3)

    start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")

    d = timedelta(hours=1)

    dic = {}
    t = start_time
    i = 0
    while t < end_time :
        str_t = datetime.strftime(t, "%Y-%m-%d %H:%M:%S")
        dic[str_t] = i
        t += d
        i += 1
        
    actual['index'] = actual['time'].map(dic)
    actual['test_id'] = actual['station_id'] + "#" + actual['index'].map(str)
    actual.drop(["time", "index"], axis=1, inplace=True)

    actual.set_index("test_id", inplace=True)

    # step 3 : caculate smape on features 
    columns = list(actual.columns)
    columns.remove("station_id")

    result = {}
    for column in columns:
        actual_list = []
        predicted_list = []
        for index in actual.index:
            if city == "bj":
                actual_element = actual.loc[index, column]
                if not pd.isna(actual_element) : 
                    predicted_element = predicted.loc[index, column]
                    actual_list.append(actual_element)
                    predicted_list.append(predicted_element)
            elif city == "ld" : 
                station = index.split("#")[0]
                if station in ld_station_list : 
                    actual_element = actual.loc[index, column]
                    # print(actual_element)
                    if is_float(actual_element) :                                  # not np.isnan(actual_element)    # pd.isna(actual_element) : 
                        # print(actual_element)
                        predicted_element = predicted.loc[index, column]
                        actual_list.append(actual_element)
                        predicted_list.append(predicted_element)   

        smape_on_column = smape(actual_list, predicted_list)
        result[column] = smape_on_column

    # print results
    print("city : %s, submit_day : %d, predicting_day : %d,%d, submit_index : %d" %(city, submit_day, submit_day+1, submit_day+2, submit_index))
    print(result)
    print("city %s overall smape %f" %(city, np.mean(list(result.values()))))  

    actual_stations = {}
    for station in set(actual['station_id'].values) : 
        df = actual[actual["station_id"]==station]
        df = df.fillna(method="ffill")
        df.drop(["station_id"], axis=1, inplace=True)
        actual_stations[station] = df

    predicted_stations = {}
    for station in set(predicted['station_id'].values) : 
        df = predicted[predicted["station_id"]==station]
        df = df.fillna(method="ffill")
        df.drop(["station_id"], axis=1, inplace=True)
        predicted_stations[station] = df

    for station in actual_stations.keys():
        if city == "bj":
            actual_station = actual_stations[station]
            predicted_station = predicted_stations[station]
            for feature in actual_station.columns :
                actual_feature = actual_station[column].values
                predicted_feature = predicted_station[column].values

                fig = plt.figure()
                ax = fig.add_subplot(111)
                fig.suptitle("%s, %s" %(station, feature))
                ax.plot(actual_feature, label="actual data");
                ax.plot(predicted_feature, label="predicted data")
                ax.legend();
        elif city == "ld":
            if station in ld_station_list : 
                actual_station = actual_stations[station]
                predicted_station = predicted_stations[station]
                for feature in actual_station.columns :
                    actual_feature = actual_station[column].values
                    predicted_feature = predicted_station[column].values

                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    fig.suptitle("%s, %s" %(station, feature))
                    ax.plot(actual_feature, label="actual data");
                    ax.plot(predicted_feature, label="predicted data")
                    ax.legend();


def is_float(s):
    '''
    wether a string can be converted to a float.
    '''
    target_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "."]
    s = str(s)
    for i in s :
        if i not in target_list :
            return False
    return True