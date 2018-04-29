Data for this project can be found [here(password:my70)](https://pan.baidu.com/s/15q48jFovG4-s3y_lzeea5Q). Place the data folder in the root directory of the repo.

The home page of this project is [here](https://www.notion.so/tianxingye/KDD-Cup-2018-eba62397b4b5403297826b928f3fe42c), feel free to leave comments.

# ChangeLog

- 20180407 v0 初始数据分析

- 20180408 v1
  - ~~增加了缺失值处理，处理方式暂为 `ffill`~~
  - 生成了对应模型输入的训练数据
  - ~~参照了 Attention model (machine translation) 创建了一个 Attention model。当前模型在训练时每一个 epoch 的起始损失和中止时候损失一样，应该是模型定义的问题。~~
  - 新增 home page

- 20180414 v2 
  - 增加了 lstm 模型

- 20180416 v3
  - 增加了单变量和多变量 seq2seq 模型

- 20180419 v4
  - 实现了 lstm 和 seq2seq 的多变量模型，模型仅考虑空气质量数据

- 20180424 v5.1
  - 天气数据
    - 将天气数据增加到 input_feature 中
    - 针对某空气质量站点，使用距离该站点最近的网格天气站点数据
  - 空气质量数据缺失值的处理
    - 数据的载入与数据处理流程分离
      - util.data_util.parse_bj_aq_data -> util.data_util_load_bj_aq_data
    - 对空气质量数据的不同缺失情况进行了相应的处理
    - 建立 .csv 文件用于存放处理后的数据

- 20180425 v5.2

  - 数据
    - 完成了北京市
      - [空气质量数据探索](https://github.com/txytju/air-quality-prediction/blob/master/aq_data_exploration.ipynb)
      - [空气质量数据预处理](https://github.com/txytju/air-quality-prediction/blob/master/aq_data_preprocess.ipynb)
      - [天气数据探索](https://github.com/txytju/air-quality-prediction/blob/master/weather_data_exploration.ipynb)
      - [天气数据预处理](https://github.com/txytju/air-quality-prediction/blob/master/weather_data_preprocess.ipynb)
    - 在数据处理之后，进行中间数据的保存；及训练/验证集的划分
  - [模型数据生成函数](https://github.com/txytju/air-quality-prediction/blob/master/generate_data.ipynb)
    - 包括训练样本的生成和验证集的生成
    - 但是在这种数据生成方式中，对数据的归一化还没有做
  - 模型
    - 基于上述数据生成方式训练`seq2seq`模型，尚未收敛

- 20180426-27 v5.3

  - ~~数据的归一化~~
    - ~~分别在空气质量数据和天气数据的预处理文档中实现了数据的归一化~~
    - ~~将数据统计特征和正则化数据保存成中间文件~~
    - ~~后续在预测和计算指标时需要再次用到数据统计特征~~
  - ~~模型的评价~~
    - 在训练过程中，将不同时刻的模型状态保存，在完成全部训练后，对上述每一个状态的模型在验证集上进行评估
  - 超参数的选择
    - ~~应该以哪一时刻的指标来去评判？~~
      - ~~多次迭代的收敛到的指标~~
      - 训练过程中的最好的指标
    - ~~循环次数~~
      - 对原始模型训练1000次的结果表明，在 1e-3 的学习率下，模型在500次迭代以内就能达到在验证集上的最好效果(0.656)
      - 但是模型在100次迭代之后往往可以收敛到一个值，这个值高于最好效果，说明模型过拟合了
      - 使用400次迭代足够达到最好表现
    - 要调整的超参数包括
      - ~~是否使用 meo~~
        - ~~使用 meo 的效果要好于不使用~~
      - ~~use_day (bool) set to True~~
        - ~~True 只使用整天的数据，False 使用所有时间点的数据~~
        - ~~True 数据数量少，可能会导致过拟合；False数据多，不容易过拟合，但是会失去“整天信息”~~
        - ~~测试结果表明，使用整天的数据结果更好~~
      - ~~**pre_days** = [3,5,6,7]~~
        - 用来预测的天数
        - 根据 grid search 的结果，选择 3 5 6 7
          - 针对每一种选一个最好的 learning rate
      - batch_size=128
      - ~~hidden_dim = 128~~
        - 对于单站点模型的计算结果表明，hidden_dim=8 效果明显好于 hidden_dim =32，因此对于多站点模型来说，也要尝试使用更少的 hidden_dim。
        - 128, 192, 256, 320, 384, 448, 512 效果都差不多
        - 16, 32, 64 效果不理想
        - 选1个：**128**, ~~192, 256~~
      - num_stacked_layers = 3
      - lambda_l2_reg=0.003
      - GRADIENT_CLIPPING=2.5
      - KEEP_RATE = 0.5
  - 实现`seq2aeq`模型的不同版本
    - 考虑的因素包括
      - 站点
        - 多站点模型(35个站点数据同时使用)
        - 单站点模型(每个站点仅预测自己站点的数据)
      - 输入的数量：空气质量特征/天气特征
        - 只使用空气质量特征
        - 同时使用空气质量特征和天气特征
      - 输出的数量：单任务学习还是多任务学习
    - 模型包括
      - ~~每个站点仅预测自己的、一个特征(单任务学习) : 35(站点) * 3(特征) * 2(是否使用天气数据)~~
        - `train_one_station_one_feature_seq2seq.py`
        - ~~单站点模型使用更小的学习率？~~
          - ~~效果不好~~
        - ~~使用空气质量特征~~
        - ~~同时使用空气质量特征和天气特征~~
      - ~~每个站点仅预测自己的三个特征(同一个站点内部的多任务学习) : 35(站点) * 1(特征) * 2(是否使用天气数据)~~
        - ~~使用空气质量特征~~
        - ~~同时使用空气质量特征和天气特征~~
      - ~~所有站点，仅预测一个特征(不同站点在同一个特征上的多任务学习) : 1(站点) * 3(特征) * 2(是否使用天气数据)~~
        - ~~使用空气质量特征~~
        - ~~同时使用空气质量特征和天气特征~~
      - 所有站点，预测所有特征(不同站点在不同特征上的多任务学习) : 1(站点) * 1(特征) * 2(城市数量)
  - 实现与上述不同版本模型相适应的模型评价流程
    - 自动化
  - 在上述模型中找出最好的一个模型，调超参数
    - 基于训练时间的考虑，使用所有站点、预测所有特征的模型

- 20180428-29 

  - 使用全站模型调优
    - 调整在不同 pre_days 中的
  - gap(指X和y之间的间隔有多大)
    - 因为当天提交对后续两天的预测，同时由于每天可以提交三次数据，于是就
      - 凌晨中午晚上各提交一次 (假设当天为0天，后边为正数天，前边为负数天)
        - 凌晨时候使用 -n~-1 day数据预测1,2 day数据
        - 中午使用-n~-1 day + 0day(0h~12h) 预测1,2 day数据
        - 晚上11点以后，使用(-n~0) day 预测1,2 day数据
  - ~~使用多种回归的损失函数~~
    - ~~L2~~
    - ~~L1~~
    - ~~Huber_loss~~
  - 伦敦数据下载整理
    - 伦敦提供了24个空气质量站点的数据，需要预测的是其中的13个
    - 伦敦也提供了很多网格天气站点，使用距离空气质量站点最近的网格天气站点的数据
    - 和北京使用同样的策略，使用13个站点的空气质量数据和天气数据
  - 提交预测数据
  - 数据
    - 实时获取的天气数据和空气质量数据的时效性如何
      - 整小时时刻之后能否立即获得数据？获得数据之后还是否有足够的时间计算？
        - aq
          - ld 延迟两个小时
    - 更新至最新数据
      - Bj 
        - aq
        - Meo
  - pipeline
    - 伦敦空气质量的预处理流程需要检查

  #  Workflow

  ​

  [workflow](https://github.com/txytju/air-quality-prediction/blob/master/project_wokflow.pdf)

  # 心得体会

  - 增加代码规划时间，动手前尽量把问题想全面
  - 接口留出充足的灵活性，避免重复造工具
  - 成果结构化，方便复用；先搭框架，再调模型
  - 不要对原始数据手动处理


