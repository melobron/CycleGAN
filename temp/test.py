import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split

from models.CycleGAN import Generator
from dataloaders.Paired_Data import Paired
from utils import *

# Arguments
parser = argparse.ArgumentParser(description='Test CycleGAN')

parser.add_argument('--gpu_num', type=int, default=0)

# Training parameters
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=1)

# Dataset
parser.add_argument('--train_size', type=int, default=1000)
parser.add_argument('--domain1', type=str, default='Metfaces')
parser.add_argument('--domain2', type=str, default='Cartoonfaces')

# Transformations
parser.add_argument('--resize', type=bool, default=True)
parser.add_argument('--crop', type=bool, default=False)
parser.add_argument('--patch_size', type=int, default=229)
parser.add_argument('--flip_rotate', type=bool, default=False)
parser.add_argument('--normalization', type=bool, default=True)

args = parser.parse_args()


def Test_CycleGAN(args):
    device = torch.device('cuda:{}'.format(args.gpu_num))
    print(device)

    netG_A2B = Generator().to(device)
    netG_B2A = Generator().to(device)

    netG_A2B_name = 'netG_A2B_train_size_{}_{}epochs.pth'.format(args.train_size, args.n_epochs)
    netG_B2A_name = 'netG_B2A_train_size_{}_{}epochs.pth'.format(args.train_size, args.n_epochs)

    dataset_name = '{}2{}_with_buffer'.format(args.domain1, args.domain2)
    netG_A2B.load_state_dict(torch.load('./checkpoints/{}/{}'.format(dataset_name, netG_A2B_name), map_location=device))
    netG_B2A.load_state_dict(torch.load('./checkpoints/{}/{}'.format(dataset_name, netG_B2A_name), map_location=device))

    netG_A2B.eval()
    netG_B2A.eval()

    transform = A.Compose(transform_list(args), additional_targets={'image2': 'image'})
    test_dataset = Paired(domain1=args.domain1, domain2=args.domain2, train=False, transform=transform)
    _, test_dataset = train_test_split(test_dataset, test_size=10, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    save_dir = './results/{}/train_size_{}_{}epochs'.format(dataset_name, args.train_size, args.n_epochs)
    fake_domain1_dir = os.path.join(save_dir, '{}'.format(args.domain1))
    fake_domain2_dir = os.path.join(save_dir, '{}'.format(args.domain2))
    if not os.path.exists(fake_domain1_dir):
        os.makedirs(fake_domain1_dir)
    if not os.path.exists(fake_domain2_dir):
        os.makedirs(fake_domain2_dir)

    print('start image generation')
    for index, (real_domain1, real_domain2) in enumerate(test_dataloader):
        real_domain1, real_domain2 = real_domain1.to(device), real_domain2.to(device)

        fake_domain1 = 0.5 * (netG_B2A(real_domain2) + 1.0)
        fake_domain2 = 0.5 * (netG_A2B(real_domain1) + 1.0)
        real_domain1 = 0.5 * (real_domain1 + 1.0)
        real_domain2 = 0.5 * (real_domain2 + 1.0)

        one2two = torch.cat([real_domain1, fake_domain2], dim=3)
        two2one = torch.cat([real_domain2, fake_domain1], dim=3)

        save_image(one2two, os.path.join(fake_domain2_dir, '{}.png'.format(index+1)))
        save_image(two2one, os.path.join(fake_domain1_dir, '{}.png'.format(index+1)))


if __name__ == "__main__":
    Test_CycleGAN(args=args)
