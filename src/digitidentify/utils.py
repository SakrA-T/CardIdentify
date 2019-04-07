import numpy as np
import matplotlib.pyplot as plt
import keras
import os

# 数字列表
digit_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '']

# 常量
im_h = 46       # 图片高度
im_w = 120      # 图片宽度
im_chan = 3     # 图片通道数目
im_n = 4        # 每张图片分割的子图数目
w = 30          # 分割图片宽度，为im_w的四分之一


def load_img(path):
    """
    加载某张图片，每张分割成四张图片，返回x数组
    :param path: 图片路径
    :return:
    """
    im = plt.imread(path)
    im.resize(im_h, im_w, im_chan)       # 压缩成标准尺寸
    x = np.zeros((im_n, im_h, w, im_chan))        # 分割成四张图片
    for i in range(im_n):
        x[i] = im[:, i*w:(i+1)*w, :]
    return x


def load_dir_img(path):
    """
    加载目录下所有的图片，每张分割成四张图片，返回x数组和y标签数组
    :param path:
    :return:
    """
    files = os.listdir(path)
    x = np.zeros((len(files)*im_n, im_h, w, im_chan))
    y = np.zeros(len(files)*im_n)
    for i in range(len(files)):
        file_path = os.path.join(path, files[i])
        im = plt.imread(file_path)

        labels = []
        for k in files[i][0:im_n]:
            if k == '_':
                labels.append(-1)
            else:
                labels.append(int(k))

        for j in range(im_n):
            x[i*im_n+j] = im[:, j*w:(j+1)*w, :]
            y[i*im_n+j] = labels[j]

    return x, y


def load_data(train_path='../data/train/', test_path='../data/test/'):
    """
    加载训练集和测试集
    :param train_path:
    :param test_path:
    :return:
    """
    return load_dir_img(train_path), load_dir_img(test_path)


def to_digit(cate):
    """
    y标签转化为对应的数字
    :param cate:
    :return:
    """
    return digit_list[np.argmax(cate)]


def img_2_digits(path):
    """
    使用模型预测图片中的数字，返回一个对应的数字字符串
    :param path: 图片路径
    :return:
    """
    im = load_img(path)
    x = im.reshape(im.shape[0], -1)     # 拉平矩阵
    model = keras.models.load_model('model2.h5')       # 加载训练好的模型
    y = model.predict(x)        # 预测
    res = ''
    for i in range(y.shape[0]):
        res += to_digit(y[i])

    return res


if __name__ == '__main__':
    # img_path = '../data/test/_091a_0.png'
    img_path = '../data/test/_081b_0.png'
    plt.imshow(plt.imread(img_path))
    plt.show()
    print(img_2_digits(img_path))
