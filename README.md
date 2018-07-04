## 1. Introduction

- [Kdd cup](https://biendata.com/competition/kdd_2018/) data mining competition, the main task is to predict air quality(aq) in Beijing and London in the next 48 hours. 
- Use seq2seq and xgboost models, ranking 31th in the final [leaderboard](https://biendata.com/competition/kdd_2018/ranking_list/).

## 2. Data Exploration and Preprocess

### 2.1 Exploratory analysis of data

- The [spatial distribution](https://github.com/txytju/air-quality-prediction/blob/master/exploration/bj_weather_data_exploration.ipynb)of the sites.
- [Correlation analysis](https://github.com/txytju/air-quality-prediction/blob/master/exploration/bj_aq_data_exploration.ipynb) data of different sites.
- [Clustering](https://github.com/txytju/air-quality-prediction/blob/master/exploration/clusting.ipynb) of different kinds of stations in Beijing.

### 2.2 Data Preprocess

[Data preprocess](https://github.com/txytju/air-quality-prediction/blob/master/data_preprocess.py) then split the dataset into training, val and aggr dataset.

1. Data preprocess

   Steps of data preprocess:

   1. Remove duplicated data. Some of the hour data are duplicated, remove them.
   2. Missing value processing. If hour level data are missing for all stations for 5 hours in a row, all (X,y) data that have these missing data in X or y are droped. Then if data are missing for all stations for less than 5 hours in a row, data before and after missing data are used to generate padding data linearly. In some cases, data for some specific stations are nan, then data from the nearest station will be used to pad.

2. Split the data

   All data points that are valid after data preprocess will be split into 3 parts : training set, validation set and aggregation set. 

   Training set is used for the training of single models, and usually data from 20170101-20180328 will be used in the training set. 

   Validation set will be used for selecting the best single models from the checkpoints of all single models. Then all best single models will be aggregated on the validation set and eveluated finally on the aggregation set. The aggregation model will be used for the final prediction.

### 2.3 [Oversampling](https://github.com/txytju/air-quality-prediction/blob/master/model/seq2seq/seq2seq_data_util.py)

1. Why oversampling?

- [Symmetric mean absolute percentage error (SMAPE)](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error) is used in this competation as evaluation metric. In SMAPE, relative error matters rather than absolut error, as shown in the function.

![](http://p3rz3gu1u.bkt.clouddn.com/2018-07-02-Screen%20Shot%202018-07-02%20at%205.20.26%20PM.png)

- However, loss functions like L1 loss, L2 loss and huber loss are applied in different models and they all aim at decreasing absolute error rather than relative error. So if models are trained using original data and these 3 loss functions, trained models would be optimized to fit data points with huge number rather than data points with smaller numbers, which would lead to larger SAMPE when evaluating with validation set and test with test set.

2. Oversamping Strategies

- Training data from 20170101-20180328 are used in the training data. Steps are as follows:

  1. PM2.5 mean of y is caculated for every (X,y) pair, and all data points in the training set are sorted in ascending order. 
  2. Smallest **oversample_part** of all datapoints are picked and repeated for **repeats** times and are appended to the original training dataset. So (1+**repeats*oversample_part**) times the original amount of training data are finally used to generate training data batch (X,y), which may help to shift the optimization target from those loss functions to SMAPE. 

  **Oversample_part** and **repeats** are hyperparameters which suitable values can be found by random search or grid search.

## 3. Models

#### 3.1 seq2seq

[Seq2seq](https://github.com/google/seq2seq) model is a machine learning model that use **decoder** and **encoder** to learn serialized feature pattern from data. Seq2seq model is applied to a lot of machine learning applications, especially NLP applications like Machine translation. In this project, seq2seq is applied to generate time series forecast of different granularity, which are **Day model** and **Hour model**. The basic graph of seq2seq model is as follows.

![](http://p3rz3gu1u.bkt.clouddn.com/2018-07-02-105238.png)

1. Day model

   The air condition seem to be very cyclical every day, as shown in the 3rd part in  [bj_aq_data_exploration](https://github.com/txytju/air-quality-prediction/blob/master/exploration/bj_aq_data_exploration.ipynb) and below. So the basic seq2seq model would be **Day model**, which means that we just predict the mean value of all aq parameters in the next 2 days, and then overlay the parameter trend during 24 hours to generate the final prediction.

   |                            PM2.5                             |                             PM10                             |                              O3                              |                             NO2                              |
   | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
   | <p float="left"> <img src="http://p3rz3gu1u.bkt.clouddn.com/2018-07-02-104135.png" width="200" />  </p> | <p float="left"> <img src="http://p3rz3gu1u.bkt.clouddn.com/2018-07-02-104144.png" width="200" />  </p> | <p float="left"> <img src="http://p3rz3gu1u.bkt.clouddn.com/2018-07-02-104219.png" width="200" />  </p> | <p float="left"> <img src="http://p3rz3gu1u.bkt.clouddn.com/2018-07-02-104240.png" width="200" />  </p> |

   The computation graph of Day model is as follws.
   

2. Hour model, Predicting 2 days together

3. Hour model, Predicting 1 day at a time

### 3.2 xgboost

### 3.3 models aggregation


