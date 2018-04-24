import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt

def load_bj_aq_data():

	# load csvs 
	bj_aq_dataset_1 = pd.read_csv("./KDD_CUP_2018/Beijing/aq/beijing_17_18_aq.csv")
	bj_aq_dataset_2 = pd.read_csv("./KDD_CUP_2018/Beijing/aq/beijing_201802_201803_aq.csv")
	bj_aq_dataset_3 = pd.read_csv("./KDD_CUP_2018/Beijing/aq/new.csv")
	bj_aq_dataset = pd.concat([bj_aq_dataset_1, bj_aq_dataset_2, bj_aq_dataset_3], ignore_index=True)
	# bj_aq_dataset = pd.concat([bj_aq_dataset_1, bj_aq_dataset_2], ignore_index=True)

	# turn date from string type to datetime type
	bj_aq_dataset["time"] = pd.to_datetime(bj_aq_dataset['utc_time'])
	bj_aq_dataset.set_index("time", inplace=True)
	bj_aq_dataset.drop("utc_time", axis=1, inplace=True)

	# names of all stations
	stations = set(bj_aq_dataset['stationId'])

	# a dict of station aq, Beijing
	bj_aq_stations = {}
	for station in stations:
		bj_aq_station = bj_aq_dataset[bj_aq_dataset["stationId"]==station].copy()
		bj_aq_station.drop("stationId", axis=1, inplace=True)

		# rename
		original_names = bj_aq_station.columns.values.tolist()
		names_dict = {original_name : station+"_"+original_name for original_name in original_names}
		bj_aq_station_renamed = bj_aq_station.rename(index=str, columns=names_dict)
              
		bj_aq_stations[station] = bj_aq_station_renamed


	# merge data of different stations into one df
	bj_aq_stations_merged = pd.concat(list(bj_aq_stations.values()), axis=1)
	# add a column of (0,1,2,3,...) to count
	length = bj_aq_stations_merged.shape[0]
	order = range(length)
	bj_aq_stations_merged['order'] = pd.Series(order, index=bj_aq_stations_merged.index)

	return bj_aq_dataset, stations, bj_aq_stations, bj_aq_stations_merged


def data_perprocess():

	# NaN in dataset
	# pm25_nan = sum(bj_aq_dataset["PM2.5"].isnull())
	# pm10_nan = sum(bj_aq_dataset["PM10"].isnull())
	# no2_nan = sum(bj_aq_dataset["NO2"].isnull())
	# co_nan = sum(bj_aq_dataset["CO"].isnull())
	# o3_nan = sum(bj_aq_dataset["O3"].isnull())
	# so2_nan = sum(bj_aq_dataset["SO2"].isnull())
	# num_rows = bj_aq_dataset.shape[0]

	# print("NaN in PM2.5 is %d, %7.6f %%" %(pm25_nan, 100 * pm25_nan/num_rows))
	# print("NaN in PM10 is %d, %7.6f %%" %(pm10_nan, 100 * pm10_nan/num_rows))
	# print("NaN in NO2 is %d, %7.6f %%" %(no2_nan, 100 * no2_nan/num_rows))
	# print("NaN in CO is %d, %7.6f %%" %(co_nan, 100 * co_nan/num_rows))
	# print("NaN in O3 is %d, %7.6f %%" %(o3_nan, 100 * o3_nan/num_rows))
	# print("NaN in SO2 is %d, %7.6f %%" %(so2_nan, 100 * so2_nan/num_rows))

	return










def generate_model_data(merged_data, m, X_hours, Y_hours = 48, step=1):
	'''
        Generate all training data at a time. 
        If batch_size=1, retrun X_dataset as list of (Tx, feature_length) and Y_dataset as list of (Ty, feature_length)
        If batch_size>1, retrun X_dataset as list of (m, Tx, feature_length) and Y_dataset as list of (m, Ty, feature_length)

	Input: 
		step : sample step.
		m : batch size.
		X_hours : use how many hours in a X data.
	Return:
		list of m batch size data.
	'''
	
	X_dataset = []
	Y_dataset = []

	model_length = X_hours + Y_hours

	data_length = merged_data.shape[0]

	for i in range(0, data_length - model_length, step):
		X = merged_data.ix[i : i+X_hours].values
		Y = merged_data.ix[i+X_hours : i+model_length].values

		if m!=1 :
			X = np.expand_dims(X, axis=0) # (1, Tx, feature_length)
			Y = np.expand_dims(Y, axis=0) # (1, Ty, feature_length)
										  # otherwise not using mini-batch
										  # (Tx, feature_length), (Ty, feature_length)
		
		# 剔除 NaN
		if True in np.isnan(X) or True in np.isnan(Y):
			continue
		else : 
			X_dataset.append(X) 
			Y_dataset.append(Y)


	# if not using mini_batch, just return X_dataset and Y_dataset
	if m==1 :
		return X_dataset, Y_dataset

	# if using mini_batch, create X_batches and Y_batches
	X_batches = []
	Y_batches = []
	batch_num = len(X_dataset) // m
	for j in range(batch_num):
		X_batch = X_dataset[j*m:(j+1)*m]
		Y_batch = Y_dataset[j*m:(j+1)*m]
		X_batch = np.concatenate((X_batch), axis=0)
		Y_batch = np.concatenate((Y_batch), axis=0)
		X_batches.append(X_batch)
		Y_batches.append(Y_batch)

	return X_batches, Y_batches


# def generate_model_data_v1(merged_data, step):
# 	'''
# 	Input:
# 		step : sample step.
# 		m : batch size.
# 	Return:
# 		Data of shape (m, Tx, feature_length), (m, Ty, feature_length)
# 	'''

# 	X_dataset = []
# 	Y_dataset = []

# 	model_length = 7 * 24
# 	data_length = merged_data.shape[0]

# 	for i in range(0,data_length - model_length, step):
# 		X = merged_data.ix[i:i+5*24].values
# 		Y = merged_data.ix[i+5*24:i+7*24].values
# 		X = np.expand_dims(X, axis=0) # (1, Tx, feature_length)
# 		Y = np.expand_dims(Y, axis=0) # (1, Ty, feature_length)

# 		X_dataset.append(X) 
# 		Y_dataset.append(Y)

# 	X_batches = np.concatenate((X_dataset), axis=0)
# 	Y_batches = np.concatenate((Y_dataset), axis=0)

# 	return X_batches, Y_batches




