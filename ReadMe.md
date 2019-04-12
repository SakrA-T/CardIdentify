## 基于深度学习的银行卡号识别系统
[赛事链接](http://www.cnsoftbei.com/bencandy.php?fid=155&aid=1691)

### 数据增强

利用数据增强技术将官方提供的每一张图片扩充为80张图片  
最后将所有的图片转化为灰度图，为模型提供训练集和测试集

- 缩放加旋转
- 样本均值置零
- 剪切变换
- 通道偏移
- 缩放
- 旋转
- 缩放加通道偏移
- 旋转加通道偏移

### 卡号定位

### 数字识别
利用神经网络识别数字  
参考：手写体数字识别
#### mlp
Test accuracy: 0.96
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 512)               707072
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 512)               262656
_________________________________________________________________
dropout_2 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_3 (Dense)              (None, 11)                5643
=================================================================
```

#### cnn
Test accuracy: 0.998
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 44, 28, 32)        320
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 42, 26, 64)        18496
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 21, 13, 64)        0
_________________________________________________________________
dropout_1 (Dropout)          (None, 21, 13, 64)        0
_________________________________________________________________
flatten_1 (Flatten)          (None, 17472)             0
_________________________________________________________________
dense_1 (Dense)              (None, 128)               2236544
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0
_________________________________________________________________
dense_2 (Dense)              (None, 11)                1419
=================================================================
```