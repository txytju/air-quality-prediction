Data for this project can be found [here(password:my70)](https://pan.baidu.com/s/15q48jFovG4-s3y_lzeea5Q). Place the data folder in the root directory of the repo.

The home page of this project is [here](https://www.notion.so/tianxingye/KDD-Cup-2018-eba62397b4b5403297826b928f3fe42c), feel free to leave comments.

# ChangeLog

- 20180407 v0 初始数据分析
- 20180408 v1
  - 增加了缺失值处理，处理方式暂为 `ffill`
  - 生成了对应模型输入的训练数据
  - 参照了 Attention model (machine translation) 创建了一个 Attention model。当前模型在训练时每一个 epoch 的起始损失和中止时候损失一样，应该是模型定义的问题。
  - 新增 home page
- 20180414 v2 
  - 增加了 lstm 模型
- 20180416 v3
  - 增加了单变量和多变量 seq2seq 模型
- 20180419 v4
  - 实现了 lstm 和 seq2seq 的多变量模型，模型仅考虑空气质量数据
- 20180424 v5
  - 天气数据
    - 将天气数据增加到 input_feature 中
    - 针对某空气质量站点，使用距离该站点最近的网格天气站点数据
  - 空气质量数据缺失值的处理
    - 数据的载入与数据处理流程分离
      - util.data_util.parse_bj_aq_data -> util.data_util_load_bj_aq_data
    - 对空气质量数据的不同缺失情况进行了相应的处理
    - 建立 .csv 文件用于存放处理后的数据

