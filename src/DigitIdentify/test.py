import os
import utils

root_path = '../../data/images/'

files = os.listdir(root_path)
for file in files:
    im_path = os.path.join(root_path, file)
    # model_path = 'model_mlp.h5'
    # print('label: ', file[:-4], '   digits: ', utils.img_2_digits_mlp(im_path, model_path))
    model_path = 'model_cnn.h5'
    print('label: ', file[:-4], '   digits: ', utils.img_2_digits_cnn(im_path, model_path))

