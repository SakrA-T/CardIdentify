import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
import os

# 数字列表
digit_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '']

# 常量
im_h = 46       # 图片高度
im_w = 120      # 图片宽度
im_chan = 3     # 图片通道数目
im_chan_2 = 4   # 图片通道数目
im_n = 4        # 每张图片分割的子图数目
w = 30          # 分割图片宽度，为im_w的四分之一


def rgb2gray(rgb):
    """
    将rgb图转换为灰度图
    :param rgb:
    :return:
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def rgb2gray_2(arr):
    """
    将四张子图组成的rgb图片转换成对应的灰度图组合数组
    origin: (4, 46, 30, 3)
    target: (4, 46, 30)
    :param arr:
    :return:
    """
    res = np.zeros((im_n, im_h, w))
    for i in range(im_n):
        res[i] = rgb2gray(arr[i])
    return res


def splice_img(arr):
    """
    拼接图片
    origin: (4, 43, 30, 3)
    target: (43, 120, 3)
    :param arr: 四张子图片数组
    :return:
    """
    im = np.zeros((im_h, im_w, im_chan))
    for i in range(im_n):
        im[:, i*w:(i+1)*w, :] = arr[i]
    return im


def load_img(path):
    """
    加载某张图片，每张分割成四张图片，返回x数组
    origin: (43, 120, 3)
    target: (4, 43, 30, 3)
    :param path: 图片路径
    :return:
    """
    im = plt.imread(path)
    im.resize(im_h, im_w, im_chan)       # 压缩成标准尺寸
    x = np.zeros((im_n, im_h, w, im_chan))        # 分割成四张图片
    for i in range(im_n):
        x[i] = im[:, i*w:(i+1)*w, :]
    return x


def load_img_2(path):
    """
    加载某张图片，每张分割成四张图片，返回x数组和y标签
    origin: (43, 120, 3)
    target: (4, 43, 30, 3)  (4,)
    :param path:
    :return:
    """
    im = plt.imread(path)
    im.resize(im_h, im_w, im_chan)       # 压缩成标准尺寸
    x = np.zeros((im_n, im_h, w, im_chan))        # 分割成四张图片
    for i in range(im_n):
        x[i] = im[:, i*w:(i+1)*w, :]
    labels = []
    for k in path[-11:-7]:
        if k == '_':
            labels.append(-1)
        else:
            labels.append(int(k))
    y = np.array(labels)
    return x, y


def load_dir_img(path):
    """
    加载目录下所有的灰度图片，每张分割成四张图片，返回x数组和y标签数组
    :param path:
    :return:
    """
    files = os.listdir(path)
    # x = np.zeros((len(files)*im_n, im_h, w, im_chan))
    x = np.zeros((len(files)*im_n, im_h, w))
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
            x[i*im_n+j] = im[:, j*w:(j+1)*w]
            y[i*im_n+j] = labels[j]

    return x, y


def load_data(train_path='../../data/train/', test_path='../../data/test/'):
    """
    加载训练集和测试集
    :param train_path:
    :param test_path:
    :return:
    """
    return load_dir_img(train_path), load_dir_img(test_path)


def load_data_2(img_dir='../../data/images_gray/', npz_path='../../data/cards.npz'):
    """
    加载目录下所有图片，自动分成训练集和测试集
    :param npz_path:
    :param img_dir:
    :return:
    """
    x, y = load_dir_img(img_dir)
    print('load images finished')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    np.savez(npz_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print('data saved as ', npz_path)
    return (x_train, y_train), (x_test, y_test)


def load_data_3(path='../../data/cards.npz'):
    """
    加载已保存的数据
    :param path:
    :return:
    """
    if os.path.exists(path):
        print('load saved data')
        with np.load(path) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)
    else:
        print('initial load data')
        return load_data_2(npz_path=path)


def to_digit(cate):
    """
    y标签转化为对应的数字
    :param cate:
    :return:
    """
    return digit_list[np.argmax(cate)]


def img_2_digits_mlp(im_path, model_path='model.h5'):
    """
    使用mlp模型预测图片中的数字，返回一个对应的数字字符串
    :param model_path:
    :param im_path:
    :return:
    """
    im = load_img(im_path)
    im = rgb2gray_2(im)
    x = im.reshape(im.shape[0], -1)     # 拉平矩阵
    model = keras.models.load_model(model_path)       # 加载训练好的模型
    y = model.predict(x)        # 预测
    res = ''
    for i in range(y.shape[0]):
        res += to_digit(y[i])

    return res


def img_2_digits_cnn(im_path, model_path='model_cnn.h5'):
    """
    使用cnn模型预测图片中的数字，返回一个对应的数字字符串
    :param im_path:
    :param model_path:
    :return:
    """
    im = load_img(im_path)
    im = rgb2gray_2(im)
    x = im.reshape(im.shape[0], im_h, w, 1)
    model = keras.models.load_model(model_path)       # 加载训练好的模型
    y = model.predict(x)        # 预测
    res = ''
    for i in range(y.shape[0]):
        res += to_digit(y[i])

    return res


if __name__ == '__main__':
    x, y = load_img_2('../../data/test/728_a_0.png')
    print(x)
    print(y)
