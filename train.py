## 主程序
from aq_data_preprocess import aq_data_preprocess
from ld_aq_data_preprocess import aq_data_preprocess as ld_aq_data_preprocess

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
print("Finished Beijing aq data preprocess.")
aq_data_preprocess(city='ld')
print("Finished London aq data preprocess.")
meo_data_preprocess(city='bj')
print("Finished Beijing meo data preprocess.")
meo_data_preprocess(city='ld')
print("Finished London meo data preprocess.")

# 3. 训练集验证集划分
train_dev_set_split(city="bj")
train_dev_set_split(city="ld")

# 4. 训练模型
model_preds_list = []
model_names = []
aver_smapes_bests = []


pred_days_list = [5,6,7]
loss_functions = ["L2", "L1", "huber_loss"]

for pred_days in pred_days_list :
    for loss_function in loss_functions :
        aver_smapes_best, model_preds, model_name= train_and_dev(city='bj',
                                                                 pre_days=pred_days, 
                                                                 gap=gap, 
                                                                 loss_function=loss_function)
    	model_preds_list.append(model_preds)
    	model_names.append(model_name)
    	aver_smapes_bests.append(aver_smapes_best)

# 5. 模型融合


# 6. 使用融合的模型预测



# 7. 自动提交结果

# 完成内容
# 1. 数据的加载、预处理
# 2. 训练集和验证集的划分
# 3. 使用不同的预测天数，不同的损失函数预测多个模型，并保存模型的预测结果




# TODO：
# 1. 每天定时跑，并且将时间和 gap 变量对应上
# 2. 当前 model_preds 是在正则化数据上的，将之转换到原始数据分布 statistics
# 3. UTC time

