from torch.utils.data import Dataset
import cv2
import random
from utils import *


class TranslationDataset(Dataset):
    def __init__(self, domain1, domain2, domain1_size, domain2_size, train=True, transform=None):
        super(TranslationDataset, self).__init__()

        img_dir = '../datasets/'
        domain1_dir = os.path.join(img_dir, domain1)
        domain2_dir = os.path.join(img_dir, domain2)
        if train:
            domain1_dir = os.path.join(domain1_dir, 'train')
            domain2_dir = os.path.join(domain2_dir, 'train')
        else:
            domain1_dir = os.path.join(domain1_dir, 'test')
            domain2_dir = os.path.join(domain2_dir, 'test')

        self.domain1_size = domain1_size
        self.domain2_size = domain2_size

        self.domain1_paths = sorted(make_dataset(domain1_dir))[:domain1_size]
        self.domain2_paths = sorted(make_dataset(domain2_dir))[:domain2_size]

        self.transform = transform

    def __getitem__(self, index):
        domain1_path = self.domain1_paths[random.randrange(self.domain1_size)]
        domain2_path = self.domain2_paths[random.randrange(self.domain2_size)]

        domain1_numpy = cv2.cvtColor(cv2.imread(domain1_path), cv2.COLOR_BGR2RGB)
        domain2_numpy = cv2.cvtColor(cv2.imread(domain2_path), cv2.COLOR_BGR2RGB)

        domain1_tensor = self.transform(domain1_numpy)
        domain2_tensor = self.transform(domain2_numpy)

        return {'domain1': domain1_tensor, 'domain2': domain2_tensor}

    def __len__(self):
        return max(len(self.domain1_paths), len(self.domain2_paths))
