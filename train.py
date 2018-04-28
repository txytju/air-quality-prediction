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
aq_data_preprocess(city='ld')
meo_data_preprocess(city='bj')
meo_data_preprocess(city='ld')

# 3. 训练集验证集划分
train_dev_set_split(city="bj")
train_dev_set_split(city="ld")

# 4. 训练模型
pred_days_list = [3,5,7,10]
loss_functions = ["L2", "L1", "huber_loss"]

for pred_days in pred_days_list : 
    for loss_function in loss_functions :
        model_preds, model_name = train_and_dev(city='bj', pre_days=pred_days, gap=gap, loss_function=loss_function)



# TODO：
# 1. 每天定时跑，并且将时间和 gap 变量对应上