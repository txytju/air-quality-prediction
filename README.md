## Introduction

- [Kdd cup](https://biendata.com/competition/kdd_2018/) data mining competition, the main task is to predict air quality (aq) in Beijing and London in the next 48 hours. 
- Use seq2seq and xgboost models, ranking 31th in the final [leaderboard](https://biendata.com/competition/kdd_2018/ranking_list/).

## Data

- [x] Exploratory analysis of data.
  - [x] The [spatial distribution](https://github.com/txytju/air-quality-prediction/blob/master/exploration/bj_weather_data_exploration.ipynb)of the sites.
  - [x] [Correlation analysis](https://github.com/txytju/air-quality-prediction/blob/master/exploration/bj_aq_data_exploration.ipynb)data of different sites.
  - [x] [Clustering](https://github.com/txytju/air-quality-prediction/blob/master/exploration/clusting.ipynb) of different kinds of stations in Beijing.
- [x] Data preprocess.
- [x] Split the dataset.
- [x] Oversampling.

## Models

- [x] seq2seq
  - [x] Day model
  - [x] hour model
    - [x] Predicting 2 days together
    - [x] Predicting 2 days 1 by 1
- [x] xgboost
- [x] models aggregation


