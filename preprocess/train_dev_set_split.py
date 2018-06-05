# coding: utf-8
import pandas as pd
import datetime



def train_dev_set_split(city="bj", data_type="norm") :
	'''
	dev_start_time_tuple : a tuple of (year, month, day)

	Args : 
	    data_type : which kind of data to use, "original" for original data, 
	                "norm" for standard normal distribution, "0-1" for 0-1 range data.
	'''
	if data_type == "norm" :
		meo = pd.read_csv("./preprocessed_data/before_split/%s_meo_norm_data.csv" %(city))
		aq = pd.read_csv("./preprocessed_data/before_split/%s_aq_norm_data.csv" %(city))
	elif data_type == "original" :
		meo = pd.read_csv("./preprocessed_data/before_split/%s_meo_data.csv" %(city))
		aq = pd.read_csv("./preprocessed_data/before_split/%s_aq_data.csv" %(city))		

	# meo.rename(index=str, columns={'Unnamed: 0':'time'}, inplace=True)
	# meo['time'] = pd.to_datetime(meo['time'])
	meo['date'] = pd.to_datetime(meo['date'])
	aq['time'] = pd.to_datetime(aq['time'])

	meo.set_index("date", inplace=True)
	aq.set_index("time", inplace=True)

	# print(aq.shape, meo.shape)
	# print(meo.index.min(), meo.index.max())
	# print(aq.index.min(), aq.index.max())


	# ### 1. 验证集 & aggregations
	dev_start_time = "2018-5-1 0:00"

	aq_dev = aq.loc[dev_start_time : ]
	meo_dev = meo.loc[dev_start_time : ]

	if data_type == "norm" :
		meo_dev.to_csv("preprocessed_data/after_split/norm_data/%s_meo_dev_data.csv" %(city))
		aq_dev.to_csv("preprocessed_data/after_split/norm_data/%s_aq_dev_data.csv" %(city))
	elif data_type == "original" :
		meo_dev.to_csv("preprocessed_data/after_split/original_data/%s_meo_dev_data.csv" %(city))
		aq_dev.to_csv("preprocessed_data/after_split/original_data/%s_aq_dev_data.csv" %(city))		

	# ### 2. 训练集
	# - 取2018年4月之前的所有数据作为训练集
	# - 由于两个数据集的开始时间不一致，因此统一截取 `2017/1/2 00:00` 开始计算训练集

	train_start_time = "2017/1/2 0:00"
	train_end_time = "2018/4/25 0:00"

	meo_train = meo.loc[train_start_time : train_end_time]
	aq_train = aq.loc[train_start_time : train_end_time]

	if data_type == "norm" :
		meo_train.to_csv("preprocessed_data/after_split/norm_data/%s_meo_train_data.csv" %(city))
		aq_train.to_csv("preprocessed_data/after_split/norm_data/%s_aq_train_data.csv" %(city))
	elif data_type == "original" :
		meo_train.to_csv("preprocessed_data/after_split/original_data/%s_meo_train_data.csv" %(city))
		aq_train.to_csv("preprocessed_data/after_split/original_data/%s_aq_train_data.csv" %(city))		
