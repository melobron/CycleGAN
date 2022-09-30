import os
import random

import torch
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


################################# Path & Directory #################################
def is_image_file(filename):
    extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
    return any(filename.endswith(extension) for extension in extensions)


def make_dataset(dir):
    img_paths = []
    assert os.path.isdir(dir), '{} is not a valid directory'.format(dir)

    for (root, dirs, files) in sorted(os.walk(dir)):
        for filename in files:
            if is_image_file(filename):
                img_paths.append(os.path.join(root, filename))
    return img_paths


################################# Model #################################
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


################################# Training #################################
class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        history = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                history.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    history.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    history.append(element)
        return torch.cat(history)


################################# Transforms #################################
def transform_list(args):
    transform = []
    if args.resize:
        transform.append(A.Resize(int(args.patch_size*1.12), int(args.patch_size*1.12), interpolation=cv2.INTER_CUBIC, p=1))
    if args.crop:
        transform.append(A.RandomCrop(args.patch_size, args.patch_size, p=1))
    if args.flip_rotate:
        transform.append(
            A.OneOf([
                A.HorizontalFlip(p=1),
                A.RandomRotate90(p=1),
                A.VerticalFlip(p=1)
            ])
        )
    if args.normalization:
        transform.append(
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), p=1, max_pixel_value=1.0)
        )
    transform.append(ToTensorV2())
    return transform
