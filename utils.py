import os
import torch
import random
import torchvision.transforms as transforms
from PIL import Image


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


def make_exp_dir(main_dir):
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)

    dirs = os.listdir(main_dir)
    dir_nums = []
    for dir in dirs:
        dir_num = int(dir[3:])
        dir_nums.append(dir_num)
    if len(dirs) == 0:
        new_dir_num = 1
    else:
        new_dir_num = max(dir_nums) + 1
    new_dir_name = 'exp{}'.format(new_dir_num)
    new_dir = os.path.join(main_dir, new_dir_name)
    return {'new_dir': new_dir, 'new_dir_num': new_dir_num}


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
def get_transforms(args):
    transform_list = [transforms.ToTensor()]
    if args.resize:
        transform_list.append(transforms.Resize(args.patch_size, transforms.InterpolationMode.BICUBIC))
    if args.flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    if args.normalize:
        transform_list.append(transforms.Normalize(mean=args.mean, std=args.std))
    return transform_list


def tensor_to_numpy(tensor):
    img = tensor.mul(255).to(torch.uint8)
    img = img.numpy().transpose(1, 2, 0)
    return img




