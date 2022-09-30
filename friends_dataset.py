from torch.utils.data import Dataset
import cv2
import random
from utils import *


class FriendsDataset(Dataset):
    def __init__(self, transform=None):
        super(FriendsDataset, self).__init__()

        img_dir = '../all_datasets/friends_faces/'
        self.img_paths = sorted(make_dataset(img_dir))
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_numpy = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_numpy)
        return img_tensor

    def __len__(self):
        return len(self.img_paths)
