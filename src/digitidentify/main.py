import os
from digitidentify import utils

root_path = '../data/test/'
img_name = '0_94b_0.png'

files = os.listdir(root_path)
for file in files:
    print('label: ', file[:-4], '   digits: ', utils.img_2_digits(os.path.join(root_path, file)))

