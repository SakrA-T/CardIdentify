from keras.preprocessing import image
import os
import matplotlib.pyplot as plt
import numpy as np
from imageio import imwrite

"""
数据增强实现
将 1084 张图片扩为 1084*80 张
"""

# 常量
im_h = 46       # 图片高度
im_w = 120      # 图片宽度
im_chan = 3     # 图片通道数目
im_n = 4        # 每张图片分割的子图数目
w = 30          # 分割图片宽度，为im_w的四分之一


def load_img_t(path):
    """
    加载完整的一张图片
    origin: (43, 120, 3)
    target: (1, 43, 120, 3)
    :param path: 图片路径
    :return:
    """
    im = plt.imread(path)
    im.resize(im_h, im_w, im_chan)       # 压缩成标准尺寸
    x = np.zeros((1, im_h, im_w, im_chan))
    x[0] = im
    return x


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


im_dir = '../../data/images/'
im_aug_dir = '../../data/images_aug/'

files = os.listdir(im_dir)

n = 10

# 1 - 10
# 缩放与旋转
im_gen1 = image.ImageDataGenerator(zoom_range=[2, 1], rotation_range=40)

for file in files:
    im = load_img(os.path.join(im_dir, file))
    im_iter = im_gen1.flow(im, batch_size=4, shuffle=False)
    for i in range(n):
        imwrite(im_aug_dir + file[:-5] + str(i+1) + '.png', splice_img(im_iter.next()))
        # plt.imsave(im_aug_dir + file[:-5] + str(i+1) + '.png', splice_img(im_iter.next()))

# 11 - 20
im_gen2 = image.ImageDataGenerator(samplewise_center=True,
                                   samplewise_std_normalization=True)
for file in files:
    im = load_img(os.path.join(im_dir, file))
    im_iter = im_gen2.flow(im, batch_size=4, shuffle=False)
    for i in range(n):
        imwrite(im_aug_dir + file[:-5] + str(i+1+n) + '.png', splice_img(im_iter.next()))
        # plt.imsave(im_aug_dir + file[:-5] + str(i+1+n) + '.png', splice_img(im_iter.next()))

# 21 - 30
im_gen3 = image.ImageDataGenerator(shear_range=0.5)
for file in files:
    im = load_img(os.path.join(im_dir, file))
    im_iter = im_gen3.flow(im, batch_size=4, shuffle=False)
    for i in range(n):
        imwrite(im_aug_dir + file[:-5] + str(i+1+n*2) + '.png', splice_img(im_iter.next()))
        # plt.imsave(im_aug_dir + file[:-5] + str(i+1+n*2) + '.png', splice_img(im_iter.next()))

# 31 - 40
im_gen4 = image.ImageDataGenerator(channel_shift_range=0.2)
for file in files:
    im = load_img_t(os.path.join(im_dir, file))
    im_iter = im_gen4.flow(im, batch_size=1, shuffle=False)
    for i in range(n):
        imwrite(im_aug_dir + file[:-5] + str(i+1+n*3) + '.png', im_iter.next()[0])
        # plt.imsave(im_aug_dir + file[:-5] + str(i+1+n*3) + '.png', im_iter.next()[0])

# 41 - 50
im_gen5 = image.ImageDataGenerator(zoom_range=[2, 1])

for file in files:
    im = load_img(os.path.join(im_dir, file))
    im_iter = im_gen5.flow(im, batch_size=4, shuffle=False)
    for i in range(n):
        imwrite(im_aug_dir + file[:-5] + str(i+1+n*4) + '.png', splice_img(im_iter.next()))
        # plt.imsave(im_aug_dir + file[:-5] + str(i+1+n*4) + '.png', splice_img(im_iter.next()))

# 51 - 60
im_gen6 = image.ImageDataGenerator(rotation_range=40)

for file in files:
    im = load_img(os.path.join(im_dir, file))
    im_iter = im_gen6.flow(im, batch_size=4, shuffle=False)
    for i in range(n):
        imwrite(im_aug_dir + file[:-5] + str(i+1+n*5) + '.png', splice_img(im_iter.next()))
        # plt.imsave(im_aug_dir + file[:-5] + str(i+1+n*5) + '.png', splice_img(im_iter.next()))

# 61 - 70
im_gen7 = image.ImageDataGenerator(zoom_range=[1.5, 1], channel_shift_range=0.1)

for file in files:
    im = load_img(os.path.join(im_dir, file))
    im_iter = im_gen7.flow(im, batch_size=4, shuffle=False)
    for i in range(n):
        imwrite(im_aug_dir + file[:-5] + str(i+1+n*6) + '.png', splice_img(im_iter.next()))
        # plt.imsave(im_aug_dir + file[:-5] + str(i+1+n*6) + '.png', splice_img(im_iter.next()))

# 71 - 80
im_gen8 = image.ImageDataGenerator(rotation_range=30, channel_shift_range=0.1)

for file in files:
    im = load_img(os.path.join(im_dir, file))
    im_iter = im_gen8.flow(im, batch_size=4, shuffle=False)
    for i in range(n):
        imwrite(im_aug_dir + file[:-5] + str(i+1+n*7) + '.png', splice_img(im_iter.next()))
        # plt.imsave(im_aug_dir + file[:-5] + str(i+1+n*7) + '.png', splice_img(im_iter.next()))


