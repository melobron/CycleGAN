import os
from utils import *
from PIL import Image
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

img_dir = '../datasets/AFAD/hi'
img_paths = make_dataset(img_dir)

save_dir = '../datasets/AFAD/test'
import random
chosen_paths = random.sample(img_paths, k=15501)
for index, p in enumerate(chosen_paths):
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    cv2.imwrite(os.path.join(save_dir, '{}.png'.format(index+5001)), img)
    print(index+1)
