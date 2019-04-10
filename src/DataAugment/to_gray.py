from imageio import imwrite
import matplotlib.pyplot as plt
import os


def rgb2gray(rgb):
    """
    将rgb图转换为灰度图
    :param rgb:
    :return:
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


im_dir = '../../data/images/'
im_aug_dir = '../../data/images_aug/'
im_gray_dir = '../../data/images_gray/'


files = os.listdir(im_aug_dir)
for file in files:
    im = plt.imread(os.path.join(im_aug_dir, file))
    imwrite(im_gray_dir + file, rgb2gray(im))
