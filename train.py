## 主程序
from aq_data_preprocess import aq_data_preprocess
from weather_data_preprocess import meo_data_preprocess
from train_dev_set_split import train_dev_set_split
from train_seq2seq import train_and_dev

# gap 时间需自动确定
# gap 作为脚本的参数
gap = 0

# 1. 自动下载数据下载并保存
# 将新增数据下载到对应文件夹的 csv 表中

# 2. 数据预处理
# 基于上述下载的数据，进行数据预处理，并将生成的中间数据保存在 csv 表格中

# aq_data_preprocess(city='bj')
print("Finished Beijing aq data preprocess.")
# aq_data_preprocess(city='ld')
print("Finished London aq data preprocess.")
# meo_data_preprocess(city='bj')
print("Finished Beijing meo data preprocess.")
# meo_data_preprocess(city='ld')
print("Finished London meo data preprocess.")

# 3. 训练集验证集划分
train_dev_set_split(city="bj")
train_dev_set_split(city="ld")

# 4. 训练模型
model_preds_on_dev = []
model_preds_on_test = []
model_names = []
aver_smapes_bests = []


pre_days_list = [5,6,7]
loss_functions = ["L2", "L1", "huber_loss"]

for pre_days in pre_days_list :
    for loss_function in loss_functions :
        print("使用%d天，使用%s损失函数" %(pre_days, loss_function))
        aver_smapes_best, model_pred_on_dev, model_pred_on_test, model_name= train_and_dev(city='bj',
                                                                                           pre_days=pre_days, 
                                                                                           gap=gap, 
                                                                                           loss_function=loss_function)
        print("使用%d天，使用%s损失函数，best_sampe = %.5f" %(pre_days, loss_function, aver_smapes_best))
        model_preds_on_dev.append(model_pred_on_dev)
        model_preds_on_test.append(model_pred_on_test)
        model_names.append(model_name)
        aver_smapes_bests.append(aver_smapes_best)

# 5. 模型融合


# 6. 使用融合的模型预测



# 7. 自动提交结果

# 完成内容
# 2. 训练集和验证集的划分
# 3. 使用不同的预测天数，不同的损失函数预测多个模型，并保存模型的预测结果




# TODO：
# 1. 每天定时跑，并且将时间和 gap 变量对应上
# 2. gap 
# 3. 生成 X test 数据集
# 4. 保证　dev　数据集上最后一天的截止时间是一致的


