from preprocess.aq_data_preprocess import aq_data_preprocess
from preprocess.meo_data_preprocess import meo_data_preprocess, pred_meo_data_preprocess
from preprocess.train_dev_set_split import train_dev_set_split

# data_preprocess
aq_data_preprocess(city='bj')
aq_data_preprocess(city='ld')
meo_data_preprocess(city='bj')
meo_data_preprocess(city='ld')
pred_meo_data_preprocess(city='bj')
pred_meo_data_preprocess(city='ld')

# split norm data
train_dev_set_split(city="bj", data_type="norm")
train_dev_set_split(city="ld", data_type="norm")
# split original data
train_dev_set_split(city="bj", data_type="original")
train_dev_set_split(city="ld", data_type="original")
