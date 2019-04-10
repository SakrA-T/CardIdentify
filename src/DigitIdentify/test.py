import os
import utils
import keras

"""
测试数字识别
"""

root_path = '../../data/images/'
im_h = 46
w = 30

model = keras.models.load_model('model_cnn.h5')

files = os.listdir(root_path)
for file in files:
    im_path = os.path.join(root_path, file)
    im = utils.load_img(im_path)
    im = utils.rgb2gray_2(im)
    x = im.reshape(im.shape[0], im_h, w, 1)
    print('label: ', file[:-4], '   digits: ', utils.img_2_digits(x, model))
    # model_path = 'model_mlp.h5'
    # print('label: ', file[:-4], '   digits: ', utils.img_2_digits_mlp(im_path, model_path))
    # model_path = 'model_cnn.h5'
    # print('label: ', file[:-4], '   digits: ', utils.img_2_digits_cnn(im_path, model_path))

