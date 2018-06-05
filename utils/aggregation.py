# load file
import pandas as pd
import numpy as np
from zoopt import Dimension, Objective, Parameter, Opt

global y_pred_L1_1
global y_pred_L1_2
global y_pred_L1_3

global y


def smape(actual, predicted):
    a = np.abs(np.array(actual) - np.array(predicted))
    b = np.array(actual) + np.array(predicted)

    return 2 * np.mean(np.divide(a, b, out=np.zeros_like(a), where=b!=0, casting='unsafe'))


def ackley(solution):
    x = solution.get_x()
    y_pred = x[0]*y_pred_L1_1 + x[1]*y_pred_L1_2 + x[2]*y_pred_L1_3 

    value = smape(y, y_pred)
    return value


def ackley2(solution):
    x = solution.get_x()
    y_pred =  x[0]*y_pred_L1_1 + x[1]*y_pred_L1_2 + x[2]*y_pred_L1_3 + x[3]

    value = smape(y, y_pred)
    return value

def ackley3(solution):
    x = solution.get_x()
    y_pred =  x[0]*y_pred_L1_1 + x[1]*y_pred_L1_2 + (1-x[0]-x[1])*y_pred_L1_3 

    value = smape(y, y_pred)
    return value

def get_feature_value_from_data(feature, datatype, folder):
    
    '''
    feature : one of ["PM2.5", "PM10", "O3"]
    datatype : one of ["dev", "aggr", "test"]
    '''
    name_1 = "%s/%s/city_%s_predays_5_L1_loss_model_preds_on_%s.csv" %(folder, city, city, datatype)
    name_2 = "%s/%s/city_%s_predays_6_L1_loss_model_preds_on_%s.csv" %(folder, city, city, datatype)
    name_3 = "%s/%s/city_%s_predays_7_L1_loss_model_preds_on_%s.csv" %(folder, city, city, datatype)

    print(name_1)
    df_y_pred_L1_1 = pd.read_csv(name_1)
    df_y_pred_L1_2 = pd.read_csv(name_2)
    df_y_pred_L1_3 = pd.read_csv(name_3)
    df_y_pred_L1_1 = select_feature_columns(df_y_pred_L1_1, feature)
    df_y_pred_L1_2 = select_feature_columns(df_y_pred_L1_2, feature)
    df_y_pred_L1_3 = select_feature_columns(df_y_pred_L1_3, feature) 
    y_pred_L1_1 = np.array(df_y_pred_L1_1)
    y_pred_L1_2 = np.array(df_y_pred_L1_2)
    y_pred_L1_3 = np.array(df_y_pred_L1_3)


    return y_pred_L1_1, y_pred_L1_2, y_pred_L1_3 


def select_feature_columns(df, feature):

    seleted_columns = [column for column in df.columns if feature in column]
    df = df[seleted_columns]

    return df



use_day_model = False
city = 'ld'
features = ["PM2.5", "PM10", "O3"]
feature = features[0]

# 1. dev set 
if use_day_model :
    folder = "model_preds_day"
    model_name = "day_model"
else :
    folder = "model_preds"
    model_name = "hour_model"


print(model_name, city, feature)

df_y = pd.read_csv("%s/%s/city_%s_dev_y.csv" %(folder, city, city))
df_y = select_feature_columns(df_y, feature)
y = np.array(df_y)

y_pred_L1_1, y_pred_L1_2, y_pred_L1_3 = get_feature_value_from_data(feature, "dev", folder)

# evaluation
loss_L1_1 = smape(y,y_pred_L1_1)
loss_L1_2 = smape(y,y_pred_L1_2)
loss_L1_3 = smape(y,y_pred_L1_3)


# zoopt

dim = 3  # dimension
obj = Objective(ackley, Dimension(dim, [[0, 1]] * dim, [True] * dim))
# perform optimization
solution = Opt.min(obj, Parameter(budget=100 * dim))
# print result
solution.print_solution()
x = solution.get_x()


# 2. aggr set
# here we get a "parameters" from solution
# use aggr to predict on aggr
df_y = pd.read_csv("%s/%s/city_%s_aggr_y.csv" %(folder, city, city))
df_y = select_feature_columns(df_y, feature)
y = np.array(df_y)

y_preds = get_feature_value_from_data(feature, "aggr", folder)


y_aggr = np.zeros_like(y).astype("float64")
for i in  range(len(y_preds)):
    y_aggr += x[i] * y_preds[i]
# y_aggr += x[3]
# y_aggr += (1-x[0]-x[1])

print("smape after aggr on dev set is :")
print(solution.get_value())
print("smape after aggr on aggr set is :")
print(smape(y,y_aggr))

with open("training_results/aggr_result.txt", "a") as f:
    f.write("%s, %s, %s\n" %(model_name, city, feature))
    f.write("smape after aggr on dev set is : %f\n" %(solution.get_value()))
    f.write("smape after aggr on aggr set is : %f\n\n" %(smape(y,y_aggr)))


# 3. test set 
df_y = pd.read_csv("submission/sample_%s_submission.csv" %(city))
df_y = select_feature_columns(df_y, feature)
y = np.array(df_y)

y_preds = get_feature_value_from_data(feature, "test", folder)


y_test = np.zeros_like(y).astype("float64")
for i in  range(len(y_preds)):
    # print(y_preds[i].shape)
    y_test += x[i] * y_preds[i]
# y_test += x[3]
# y_test += (1-x[0]-x[1])

csv_name = "%s_%s_%s.csv" %(model_name, city, feature)

np.savetxt(csv_name, y_test, delimiter=",")