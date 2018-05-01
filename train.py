## 主程序
import numpy as np
import pandas as pd
from aq_data_preprocess import aq_data_preprocess
from weather_data_preprocess import meo_data_preprocess
from train_dev_set_split import train_dev_set_split
from train_seq2seq import train_and_dev

# gap 时间需自动确定
# gap 作为脚本的参数
gap = 24

# 1. 自动下载数据下载并保存
# 将新增数据下载到对应文件夹的 csv 表中

# 2. 数据预处理
# 基于上述下载的数据，进行数据预处理，并将生成的中间数据保存在 csv 表格中

# aq_data_preprocess(city='bj')
# print("Finished Beijing aq data preprocess.")
# aq_data_preprocess(city='ld')
# print("Finished London aq data preprocess.")
# meo_data_preprocess(city='bj')
# print("Finished Beijing meo data preprocess.")
# meo_data_preprocess(city='ld')
# print("Finished London meo data preprocess.")

# 3. 训练集验证集划分
# train_dev_set_split(city="bj")
# train_dev_set_split(city="ld")

# 4. 训练模型

results = {}

# pre_days_list = [5]
# loss_functions = ["L2"]
total_iteractions = 200
pre_days_list = [5,6,7]
loss_functions = ["L2", "L1", "huber_loss"]

for city in ['ld', 'bj'] :
    results[city] = {}
    for pre_days in pre_days_list :
        for loss_function in loss_functions :
            print("city %s 使用 %d 天，使用 %s 损失函数" %(city, pre_days, loss_function))
            aver_smapes_best, model_preds_on_dev, dev_y_original, model_preds_on_test, output_features = train_and_dev(city=city,   
                                                                                                                       pre_days=pre_days, 
                                                                                                                       gap=gap, 
                                                                                                                       loss_function=loss_function,
                                                                                                                       total_iteractions=total_iteractions)
            print("city %s 使用 %d 天，使用 %s 损失函数，best_sampe = %.5f" %(city, pre_days, loss_function, aver_smapes_best))
            
            results[city][aver_smapes_best] = [model_preds_on_dev, dev_y_original, model_preds_on_test, output_features]

            print(model_preds_on_test)

# 5. 模型融合
# Todo



# 6. 使用融合的模型预测
print("---------------------------------------------")
print("Stating to find best result !")

submit = {}

for city, result_of_a_city in results.items():
    min_smape_of_a_city = min(result_of_a_city.keys())
    _, _, model_preds_on_test, output_features = result_of_a_city[min_smape_of_a_city]
    submit[city] = [np.squeeze(model_preds_on_test), output_features]



def get_value(submit, target_station, target_hour, target_feature):
    for city, [preds, output_features] in submit.items():
        for index, output_feature in enumerate(output_features) :
            features = output_feature.split("_")
            if "aq" in features :
                features.remove("aq")
            station, feature = features
            if "aq" in target_station : 
                target_station = target_station.split("_")[0]
            if target_station in station and feature == target_feature :
                return preds[target_hour,index]
    return 0


# load sample
submission = pd.read_csv("submission.csv")
submission["PM2.5"] = submission["PM2.5"].astype('float64')
submission["PM10"] = submission["PM10"].astype('float64')


# 7. generate submission and save it to file
for index in submission.index:
    # print(index)
    test_id = submission.test_id[index]
    station, hour = test_id.split("#")
    
    for feature in ["PM2.5", "PM10", "O3"]:
        # print(station, int(hour), feature)
        r = get_value(submit, station, int(hour), feature)
        print(r)      
        submission.set_value(index, feature, r)

submission.set_index("test_id", inplace=True)
submission[submission<0]=0
submission.to_csv("my_submission.csv")

# print("Submit")
# submit()
# import api_submit



# TODO：
# 1. don't use normed data!
# 3. 保证　dev　数据集上最后一天的截止时间是一致的!!!!  !!!!!!!!!
# 4. dev 尾部数据中有缺失值．查看天气数据和ａｑ数据的连接方式？为什么会产生缺失值
# delta = 0  not possible because of running time ?
# 5. 每天定时跑，并且将时间和 gap 变量对应上


