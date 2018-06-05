import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt

# def plot_station(bj_aq_stations, station, feature, minimum='2017-1-1', maxmimum='2018-3-31'):
#     data = bj_aq_stations[station].loc[minimum:maxmimum]
#     for column in data.columns : 
#         if feature in column :
#             features = data[column]
#             print(type(features))
#             print(features)
#             print(features.shape)
#             plt.figure(figsize=(20,10))
#             plt.plot(features.shape,features);


# def plot_stations(bj_aq_stations, feature, 
# 				  minimum='2017-1-1', 
# 				  maxmimum='2018-3-31'):
#     plt.figure(figsize=(20,10))
#     for station in bj_aq_stations.keys():
#     	data = bj_aq_stations[station].loc[minimum:maxmimum]
#     	features = data[feature]
#     	plt.plot(features);	

def plot_forecast_and_actual_example(test_x, test_y, final_preds, features, index, feature_index):
    '''
    Plot the forecast and actual values in the dev set to compare the difference.
    index : start index.
    feature_index : index of feature in the feature list.
    '''
    x = test_x[index,:,feature_index]
    y_p = final_preds[index,:,feature_index]
    y_t = test_y[index,:,feature_index]
    
    input_seq_len = len(x)
    output_seq_len = len(y_t)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    fig.suptitle(features[feature_index])
    ax.plot(range(input_seq_len), x, label="test x data");
    ax.plot(range(input_seq_len, input_seq_len + output_seq_len), y_t, label="test y data")
    ax.plot(range(input_seq_len, input_seq_len + output_seq_len), y_p, label="prediction y data")
    ax.legend();
