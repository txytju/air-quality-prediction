import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt


def load_bj_aq_data():

	bj_csv_list = ["./KDD_CUP_2018/Beijing/aq/beijing_17_18_aq.csv",
				   "./KDD_CUP_2018/Beijing/aq/beijing_201802_201803_aq.csv",
				   "./KDD_CUP_2018/Beijing/aq/new.csv"]

	bj_aq_datas = []

	for csv in bj_csv_list :
		bj_aq_data = pd.read_csv(csv)
		bj_aq_datas.append(bj_aq_data)

	bj_aq_df = pd.concat(bj_aq_datas, ignore_index=True)


	bj_aq_dataset, stations, bj_aq_stations, bj_aq_stations_merged = load_city_aq_data(bj_aq_df)

	return bj_aq_dataset, stations, bj_aq_stations, bj_aq_stations_merged


def load_ld_aq_data():

	# 前边的是需要预测的站点，后边的是其他辅助站点
	# 预测站点有数据缺失的问题，需要使用辅助站点的数据补齐数据
	ld_csv_list = ["./KDD_CUP_2018/London/aq/London_historical_aqi_forecast_stations_20180331.csv",
	               "./KDD_CUP_2018/London/aq/London_historical_aqi_other_stations_20180331.csv"]

	ld_aq_datas = []

	for csv in ld_csv_list :
		ld_aq_data = pd.read_csv(csv)

		# 去掉伦敦多余的数字列
		for column_name in ld_aq_data.columns :
			if 'Unnamed:' in column_name : 
				ld_aq_data.drop(column_name, axis=1, inplace=True)

		# 统一表头格式：将伦敦的表头换成北京的表头格式
		# 伦敦的表头自己也不统一
		if 'station_id' in ld_aq_data.columns :
			new_names = {'MeasurementDateGMT':'utc_time', 'station_id':'stationId'}
			ld_aq_data.rename(index=str, columns=new_names, inplace=True)
		elif 'Station_ID' in ld_aq_data.columns :  
			new_names = {'MeasurementDateGMT':'utc_time', 'Station_ID':'stationId'}
			ld_aq_data.rename(index=str, columns=new_names, inplace=True)

		ld_aq_datas.append(ld_aq_data)

	ld_aq_df = pd.concat(ld_aq_datas, ignore_index=True)

	ld_aq_dataset, stations, ld_aq_stations, ld_aq_stations_merged = load_city_aq_data(ld_aq_df)

	return ld_aq_dataset, stations, ld_aq_stations, ld_aq_stations_merged


def load_city_aq_data(aq_df):

	'''
	aq_df : a dataframe after concat.
	'''
	aq_dataset = aq_df

	# turn date from string type to datetime type
	aq_dataset["time"] = pd.to_datetime(aq_dataset['utc_time'])
	aq_dataset.set_index("time", inplace=True)
	aq_dataset.drop("utc_time", axis=1, inplace=True)

	aq_dataset = aq_dataset[pd.isnull(aq_dataset.stationId) != True]

	# names of all stations
	stations = set(aq_dataset['stationId'])
	# print(stations)

	# a dict of station aq
	aq_stations = {}
	for station in stations:
		aq_station = aq_dataset[aq_dataset["stationId"]==station].copy()
		aq_station.drop("stationId", axis=1, inplace=True)

		# rename
		original_names = aq_station.columns.values.tolist()
		names_dict = {original_name : station+"_"+original_name for original_name in original_names}
		aq_station.rename(index=str, columns=names_dict, inplace=True)
              
		aq_stations[station] = aq_station


	# merge data of different stations into one df
	aq_stations_merged = pd.concat(list(aq_stations.values()), axis=1)
	# add a column of (0,1,2,3,...) to count
	length = aq_stations_merged.shape[0]
	order = range(length)
	aq_stations_merged['order'] = pd.Series(order, index=aq_stations_merged.index)

	return aq_dataset, stations, aq_stations, aq_stations_merged



# def data_perprocess():

# 	return


# def generate_model_data(merged_data, m, X_hours, Y_hours = 48, step=1):
# 	'''
#         Generate all training data at a time. 
#         If batch_size=1, retrun X_dataset as list of (Tx, feature_length) and Y_dataset as list of (Ty, feature_length)
#         If batch_size>1, retrun X_dataset as list of (m, Tx, feature_length) and Y_dataset as list of (m, Ty, feature_length)

# 	Input: 
# 		step : sample step.
# 		m : batch size.
# 		X_hours : use how many hours in a X data.
# 	Return:
# 		list of m batch size data.
# 	'''
	
# 	X_dataset = []
# 	Y_dataset = []

# 	model_length = X_hours + Y_hours

# 	data_length = merged_data.shape[0]

# 	for i in range(0, data_length - model_length, step):
# 		X = merged_data.ix[i : i+X_hours].values
# 		Y = merged_data.ix[i+X_hours : i+model_length].values

# 		if m!=1 :
# 			X = np.expand_dims(X, axis=0) # (1, Tx, feature_length)
# 			Y = np.expand_dims(Y, axis=0) # (1, Ty, feature_length)
# 										  # otherwise not using mini-batch
# 										  # (Tx, feature_length), (Ty, feature_length)
		
# 		# 剔除 NaN
# 		if True in np.isnan(X) or True in np.isnan(Y):
# 			continue
# 		else : 
# 			X_dataset.append(X) 
# 			Y_dataset.append(Y)


# 	# if not using mini_batch, just return X_dataset and Y_dataset
# 	if m==1 :
# 		return X_dataset, Y_dataset

# 	# if using mini_batch, create X_batches and Y_batches
# 	X_batches = []
# 	Y_batches = []
# 	batch_num = len(X_dataset) // m
# 	for j in range(batch_num):
# 		X_batch = X_dataset[j*m:(j+1)*m]
# 		Y_batch = Y_dataset[j*m:(j+1)*m]
# 		X_batch = np.concatenate((X_batch), axis=0)
# 		Y_batch = np.concatenate((Y_batch), axis=0)
# 		X_batches.append(X_batch)
# 		Y_batches.append(Y_batch)

# 	return X_batches, Y_batches


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




