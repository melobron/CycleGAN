import os

import torch

from utils import *
from PIL import Image
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


a = torch.zeros(size=(2, 2))
a[0:1] = 1
print(a)


