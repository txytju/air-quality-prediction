# coding: utf-8
import pandas as pd
import datetime



def train_dev_set_split(city="bj") :
	'''
	dev_start_time_tuple : a tuple of (year, month, day)
	'''

	meo = pd.read_csv("./test/%s_meo_norm_data.csv" %(city))
	aq = pd.read_csv("./test/%s_aq_norm_data.csv" %(city))

	meo.rename(index=str, columns={'Unnamed: 0':'time'}, inplace=True)
	meo['time'] = pd.to_datetime(meo['time'])
	aq['time'] = pd.to_datetime(aq['time'])

	meo.set_index("time", inplace=True)
	aq.set_index("time", inplace=True)

	# print(aq.shape, meo.shape)
	# print(meo.index.min(), meo.index.max())
	# print(aq.index.min(), aq.index.max())


	# ### 1. 验证集
	# 取4月份的数据作为验证集，当前共23天的数据

	dev_start_time = "2018-4-01 0:00"
	# year, month, day = dev_start_time_tuple
	# dev_start_time = "%d-%d-%d 0:00" %(year, month, day)
	aq_dev = aq.loc[dev_start_time:]
	meo_dev = meo.loc[dev_start_time:]

	# print(meo_dev.shape)
	# print(aq_dev.shape)


	meo_dev.to_csv("test/%s_meo_dev_data.csv" %(city))
	aq_dev.to_csv("test/%s_aq_dev_data.csv" %(city))


	# ### 2. 训练集
	# - 取2018年4月之前的所有数据作为训练集
	# - 由于两个数据集的开始时间不一致，因此统一截取 `2017/1/2 00:00` 开始计算训练集

	train_start_time = "2017/1/2 0:00"
	train_end_time = "2018/3/31 0:00"

	meo_train = meo.loc[train_start_time : train_end_time]
	aq_train = aq.loc[train_start_time : train_end_time]

	# print(meo_train.shape)
	# print(aq_train.shape)

	meo_train.to_csv("test/%s_meo_train_data.csv" %(city))
	aq_train.to_csv("test/%s_aq_train_data.csv" %(city))
