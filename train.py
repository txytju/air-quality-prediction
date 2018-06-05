## 主程序
import sys

import numpy as np
import pandas as pd
import tkinter
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from preprocess.aq_data_preprocess import aq_data_preprocess
from preprocess.meo_data_preprocess import meo_data_preprocess
from preprocess.train_dev_set_split import train_dev_set_split
from model.seq2seq.train_seq2seq import train_model
from utils.output_util import write_value_to_csv


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
session = tf.Session(config=gpu_config)
KTF.set_session(session)



# which model to use
if sys.argv[1] == "day_model" :
    use_day_model = True
    generate_mean = True
    generate_range = False
elif sys.argv[1] == "hour_model" :
    use_day_model = False
    generate_mean = False
    generate_range = False

# use norm data or original data
if sys.argv[2] == "norm_data" :
    use_norm_data = True
    print("use_norm_data.")
elif sys.argv[2] == "original_data" :
    use_norm_data = False
    print("use_original_data.")

# how long is the gap
if sys.argv[3] == "1" :
    if use_day_model : 
        gap = 1
    else :
        gap = 24
elif sys.argv[3] == "0" :
    gap = 0

# one_day_model or two_days_model
if sys.argv[4] == "one" :
    predict_one_day = True
else :
    predict_one_day = False


# 训练模型

results = {}


total_iteractions = 100
pre_days_list = [5,6,7]
loss_functions = ["L1"] # ["L1", 'huber']


for city in ['bj','ld'] :
    results[city] = {}
    dev_y_original_flag = True 
    aggr_y_original_flag = True 

    for pre_days in pre_days_list :
        for loss_function in loss_functions :
            
            print("Use day model : %r, city %s 使用 %d 天, 使用 %s 损失函数" %(use_day_model, city, pre_days, loss_function))
            
            aver_smapes_best, model_preds_on_dev, dev_y_original, model_preds_on_aggr, aggr_y_original, model_preds_on_test, output_features = train_model(city=city,   
                                                                                                                                                           pre_days=pre_days, 
                                                                                                                                                           gap=gap, 
                                                                                                                                                           loss_function=loss_function,
                                                                                                                                                           total_iteractions=total_iteractions,
                                                                                                                                                           use_day_model=use_day_model,
                                                                                                                                                           use_norm_data=use_norm_data,
                                                                                                                                                           generate_mean=generate_mean,
                                                                                                                                                           generate_range=generate_range,
                                                                                                                                                           loss_weights=loss_weights,
                                                                                                                                                           predict_one_day=predict_one_day)
            
            if use_day_model : 
                traing_result = "Use day model, city %s 使用 %d 天, 使用 %s 损失函数, best_sampe = %.5f\n" %(city, pre_days, loss_function, aver_smapes_best)
            else :
                traing_result = "Use hour model, city %s 使用 %d 天, 使用 %s 损失函数, best_sampe = %.5f\n" %(city, pre_days, loss_function, aver_smapes_best)
            print(traing_result)
            
            # write training summary results to txt files.
            traing_result_file_name = "training_results/single_model_%s.txt" %city
            with open(traing_result_file_name, "a") as f:
                f.write(traing_result)

            # write data to file for further use            
            # file_name are like : "city_bj_predays_5_L2_loss_model_preds_on_dev"
            value_name = "model_preds_on_dev"
            file_name = "city_%s_predays_%d_%s_loss_%s" %(city, pre_days, loss_function, value_name)
            write_value_to_csv(city, file_name, model_preds_on_dev, output_features, day=use_day_model, one_day_model=predict_one_day)

            value_name = "model_preds_on_test"
            file_name = "city_%s_predays_%d_%s_loss_%s" %(city, pre_days, loss_function, value_name)  
            write_value_to_csv(city, file_name, model_preds_on_test, output_features, day=use_day_model, one_day_model=predict_one_day)

            value_name = "model_preds_on_aggr"
            file_name = "city_%s_predays_%d_%s_loss_%s" %(city, pre_days, loss_function, value_name)           
            write_value_to_csv(city, file_name, model_preds_on_aggr, output_features, day=use_day_model, one_day_model=predict_one_day)


            # save only once
            while dev_y_original_flag : 
                value_name = "dev_y"
                file_name = "city_%s_%s" %(city, value_name)
                write_value_to_csv(city, file_name, dev_y_original, output_features, day=use_day_model, one_day_model=predict_one_day)
                dev_y_original_flag = False
            
            # save only once
            while aggr_y_original_flag : 
                value_name = "aggr_y"
                file_name = "city_%s_%s" %(city, value_name)
                write_value_to_csv(city, file_name, aggr_y_original, output_features, day=use_day_model, one_day_model=predict_one_day)
                aggr_y_original_flag = False

            results[city][aver_smapes_best] = [model_preds_on_dev, dev_y_original, model_preds_on_test, output_features]
