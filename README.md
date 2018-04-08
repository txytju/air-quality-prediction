Data for this project can be found [here(password:my70)](https://pan.baidu.com/s/15q48jFovG4-s3y_lzeea5Q). Place the data folder in the root directory of the repo.

# ChangeLog

- 20180407 v0 初始数据分析
- 20180408 v1
  - 增加了缺失值处理，处理方式暂为 `ffill`
  - 生成了对应模型输入的训练数据
  - 参照了 Attention model (machine translation) 创建了一个 Attention model。当前模型在训练时每一个 epoch 的起始损失和中止时候损失一样，应该是模型定义的问题。