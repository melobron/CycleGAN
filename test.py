import argparse
import cv2
import numpy as np
import torch

from model import Generator
from datasets import TranslationDataset
from utils import *

# Arguments
parser = argparse.ArgumentParser(description='Test CycleGAN')

parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--seed', default=100, type=int)
parser.add_argument('--exp_num', type=int, default=1)

# Training parameters
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=1)

# Dataset
parser.add_argument('--domain1', type=str, default='FFHQ')
parser.add_argument('--domain2', type=str, default='Dog')
parser.add_argument('--test_size', type=int, default=100)

# Transformations
parser.add_argument('--resize', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--flip', type=bool, default=False)
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--mean', type=tuple, default=(0.5, 0.5, 0.5))
parser.add_argument('--std', type=tuple, default=(0.5, 0.5, 0.5))

opt = parser.parse_args()


def Test_CycleGAN(args):
    # Device
    device = torch.device('cuda:{}'.format(opt.gpu_num))

    # Random Seeds
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)

    netG_A2B = Generator().to(device)
    netG_B2A = Generator().to(device)

    netG_A2B.eval()
    netG_B2A.eval()

    netG_A2B.load_state_dict(torch.load('./experiments/exp{}/checkpoints/netG_A2B_{}epochs.pth'.format(
        opt.exp_num, opt.n_epochs), map_location=device))
    netG_B2A.load_state_dict(torch.load('./experiments/exp{}/checkpoints/netG_B2A_{}epochs.pth'.format(
        opt.exp_num, opt.n_epochs), map_location=device))

    test_transform = transforms.Compose(get_transforms(opt))
    test_dataset = TranslationDataset(domain1=args.domain1, domain2=args.domain2,
                                      domain1_size=opt.test_size, domain2_size=opt.test_size,
                                      train=False, transform=test_transform)

    save_dir = './results/exp{}/{}epochs'.format(opt.exp_num, args.n_epochs)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for index, data in enumerate(test_dataset):
        real_A, real_B = data['domain1'], data['domain2']
        real_A, real_B = real_A.to(device), real_B.to(device)
        real_A, real_B = torch.unsqueeze(real_A, dim=0), torch.unsqueeze(real_B, dim=0)

        fake_A, fake_B = netG_B2A(real_B), netG_A2B(real_A)

        reverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-m/s for m, s in zip(opt.mean, opt.std)], std=[1/s for s in opt.std])
        ])

        AtoB = torch.cat([real_A, fake_B], dim=3)
        BtoA = torch.cat([real_B, fake_A], dim=3)

        AtoB, BtoA = reverse_transform(AtoB), reverse_transform(BtoA)
        AtoB, BtoA = torch.squeeze(AtoB, dim=0), torch.squeeze(BtoA, dim=0)
        AtoB, BtoA = AtoB.cpu().numpy().transpose(1, 2, 0), BtoA.cpu().numpy().transpose(1, 2, 0)
        AtoB, BtoA = np.clip(AtoB, 0., 1.) * 255., np.clip(BtoA, 0., 1.) * 255.
        AtoB, BtoA = cv2.cvtColor(AtoB, cv2.COLOR_RGB2BGR), cv2.cvtColor(BtoA, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(save_dir, '{}to{}{}.png'.format(opt.domain1, opt.domain2, index+1)), AtoB)
        cv2.imwrite(os.path.join(save_dir, '{}to{}{}.png'.format(opt.domain2, opt.domain1, index+1)), BtoA)


if __name__ == "__main__":
    Test_CycleGAN(args=opt)
