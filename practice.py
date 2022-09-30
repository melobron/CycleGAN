import os
from utils import *
from PIL import Image
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

img_dir = '../all_datasets/LFW_faces/faces_ppm/'
save_dir = '../all_datasets/LFW_faces/faces/'
img_paths = glob(os.path.join(img_dir, '*.ppm'))

for index, img_path in enumerate(img_paths):
    img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img_name = '{}.png'.format(index)
    save_path = os.path.join(save_dir, img_name)
    print(img.shape)
    cv2.imwrite(save_path, img)
    break


