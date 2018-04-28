## 主程序
from aq_data_preprocess import aq_data_preprocess
from weather_data_preprocess import meo_data_preprocess
from train_dev_set_split import train_dev_set_split
from train_seq2seq import train_and_dev

# gap 时间需自动确定
gap = 0

# 1. 自动下载数据下载并保存
# 将新增数据下载到对应文件夹的 csv 表中

# 2. 数据预处理
# 基于上述下载的数据，进行数据预处理，并将生成的中间数据保存在 csv 表格中

aq_data_preprocess(city='bj')
# aq_data_preprocess(city='ld')
meo_data_preprocess(city='bj')
# meo_data_preprocess(city='ld')

# 3. 训练集验证集划分
train_dev_set_split(city="bj")
# train_dev_set_split(city="ld")

# 4. 训练模型
pred_days_list = [3,5,7,10]
loss_functions = ["L2", "L1", "huber_loss"]

for pred_days in pred_days_list :

    # pred_days 不同生成的 dev set 也不同
    test_x, test_y = generate_dev_set(city=city,
                                  station_list=station_list,
                                  X_aq_list=X_aq_list, 
                                  y_aq_list=y_aq_list, 
                                  X_meo_list=X_meo_list,
                                  pre_days=pre_days,
                                  gap=gap)
    for loss_function in loss_functions :
        model_preds, model_name = train_and_dev(city='bj', pre_days=pred_days, gap=gap, loss_function=loss_function)


# 5. 模型融合


# 6. 使用融合的模型预测



# 7. 自动提交结果






# TODO：
# 1. 每天定时跑，并且将时间和 gap 变量对应上
# 2. 当前 model_preds 是在正则化数据上的，将之转换到原始数据分布 statistics
# 3. UTC time