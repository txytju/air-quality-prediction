import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt

def parse_bj_aq_data(fill_method="ffill"):
	bj_aq_dataset_1 = pd.read_csv("./KDD_CUP_2018/Beijing/beijing_17_18_aq.csv")
	bj_aq_dataset_2 = pd.read_csv("./KDD_CUP_2018/Beijing/beijing_201802_201803_aq.csv")
	bj_aq_dataset = pd.concat([bj_aq_dataset_1, bj_aq_dataset_2], ignore_index=True)

	# 将 string 类型的日期转换为 datetime 类型
	length = bj_aq_dataset.shape[0]
	formet_time = pd.Series([datetime.datetime.strptime(bj_aq_dataset["utc_time"][i],'%Y-%m-%d %H:%M:%S') for i in range(length)])
	bj_aq_dataset["format_time"] = formet_time
	# bj_aq_dataset.set_index("format_time")


	# NaN in dataset
	pm25_nan = sum(bj_aq_dataset["PM2.5"].isnull())
	pm10_nan = sum(bj_aq_dataset["PM10"].isnull())
	no2_nan = sum(bj_aq_dataset["NO2"].isnull())
	co_nan = sum(bj_aq_dataset["CO"].isnull())
	o3_nan = sum(bj_aq_dataset["O3"].isnull())
	so2_nan = sum(bj_aq_dataset["SO2"].isnull())
	num_rows = bj_aq_dataset.shape[0]

	print("NaN in PM2.5 is %d, %7.6f %%" %(pm25_nan, 100 * pm25_nan/num_rows))
	print("NaN in PM10 is %d, %7.6f %%" %(pm10_nan, 100 * pm10_nan/num_rows))
	print("NaN in NO2 is %d, %7.6f %%" %(no2_nan, 100 * no2_nan/num_rows))
	print("NaN in CO is %d, %7.6f %%" %(co_nan, 100 * co_nan/num_rows))
	print("NaN in O3 is %d, %7.6f %%" %(o3_nan, 100 * o3_nan/num_rows))
	print("NaN in SO2 is %d, %7.6f %%" %(so2_nan, 100 * so2_nan/num_rows))


	# 所有的站的名字
	stations = set(bj_aq_dataset['stationId'])
	# type(bj_aq_data['stationId'])
	print("There are %d air quality stations in Beijing" %len(stations))
	print("\nThe stations in Beijing are:\n",stations)

	# a dict of station aq, Beijing
	bj_aq_stations = {}
	bj_aq_stations_noname = {}
	for station in stations:
		bj_aq_station = bj_aq_dataset[bj_aq_dataset["stationId"]==station]
		bj_aq_station.set_index("format_time", inplace=True)
		bj_aq_station.drop("utc_time", axis=1, inplace=True)
		bj_aq_station.drop("stationId", axis=1, inplace=True)

		# rename
		original_names = bj_aq_station.columns.values.tolist()
		names_dict = {original_name : station+"_"+original_name for original_name in original_names}
		bj_aq_station_renamed = bj_aq_station.rename(index=str, columns=names_dict)

		# fill NaN
		if fill_method == "ffill":
			bj_aq_station_renamed.fillna(method="ffill", inplace=True)

		bj_aq_stations[station] = bj_aq_station_renamed


	# 整合成一个大的 dataframe
	bj_aq_stations_merged = pd.concat(list(bj_aq_stations.values()), axis=1)

	return bj_aq_dataset, stations, bj_aq_stations, bj_aq_stations_merged

def generate_model_data(merged_data, step, m):
	'''
	Input:
		step : sample step.
		m : batch size.
	Return:
		list of m batch size data.
	'''

	X_dataset = []
	Y_dataset = []

	model_length = 7 * 24
	data_length = merged_data.shape[0]

	for i in range(0,data_length - model_length, step):
		X = merged_data.ix[i:i+5*24].values
		Y = merged_data.ix[i+5*24:i+7*24].values
		X = np.expand_dims(X, axis=0) # (1, Tx, feature_length)
		Y = np.expand_dims(Y, axis=0) # (1, Ty, feature_length)
		X_dataset.append(X) 
		Y_dataset.append(Y)

	X_batches = []
	Y_batches = []
	num = len(X_dataset) // m
	for j in range(num):
		X_batch = X_dataset[j*m:(j+1)*m]
		Y_batch = Y_dataset[j*m:(j+1)*m]
		X_batch = np.concatenate((X_batch), axis=0)
		Y_batch = np.concatenate((Y_batch), axis=0)
		X_batches.append(X_batch)
		Y_batches.append(Y_batch)

	return X_batches, Y_batches


def plot_station(bj_aq_stations, station, feature, minimum=datetime.datetime(2017,1,1), maxmimum=datetime.datetime(2018,3,30)):
    data = bj_aq_stations[station].loc[minimum:maxmimum]
    features = data[feature]
    plt.figure(figsize=(20,10))
    plt.plot(features);


def plot_stations(bj_aq_stations, feature, 
				  minimum=datetime.datetime(2017,1,1), 
				  maxmimum=datetime.datetime(2018,3,30)):
    plt.figure(figsize=(20,10))
    for station in bj_aq_stations.keys():
    	data = bj_aq_stations[station].loc[minimum:maxmimum]
    	features = data[feature]
    	plt.plot(features);	