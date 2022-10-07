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
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--exp_num', type=int, default=7)

# Training parameters
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=1)

# Dataset
parser.add_argument('--domain1', type=str, default='Dog')
parser.add_argument('--domain2', type=str, default='AFAD')
parser.add_argument('--test_size', type=int, default=300)

# Transformations
parser.add_argument('--resize', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--flip', type=bool, default=False)
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--mean', type=tuple, default=(0.5, 0.5, 0.5))
parser.add_argument('--std', type=tuple, default=(0.5, 0.5, 0.5))

opt = parser.parse_args()


def reverse(args, tensor):
    reverse_transform = transforms.Compose([
        transforms.Normalize(mean=[-m / s for m, s in zip(args.mean, args.std)], std=[1 / s for s in args.std])
    ])

    out = reverse_transform(tensor)
    out = torch.squeeze(out, dim=0)
    out = out.cpu().numpy().transpose(1, 2, 0)
    out = np.clip(out, 0., 1.) * 255.
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out


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

    testA_dir = '../datasets/{}/test'.format(opt.domain1)
    testB_dir = '../datasets/{}/test'.format(opt.domain2)
    testA_paths = make_dataset(testA_dir)
    testB_paths = make_dataset(testB_dir)

    save_dir = './experiments/exp{}/results/'.format(opt.exp_num)
    AtoB_dir = os.path.join(save_dir, '{}to{}'.format(opt.domain1, opt.domain2))
    BtoA_dir = os.path.join(save_dir, '{}to{}'.format(opt.domain2, opt.domain1))
    fakeA_dir = os.path.join(save_dir, 'fake_{}'.format(opt.domain1))
    fakeB_dir = os.path.join(save_dir, 'fake_{}'.format(opt.domain2))

    for d in [AtoB_dir, BtoA_dir, fakeA_dir, fakeB_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    with torch.no_grad():
        for index, p in enumerate(testA_paths):
            real_A = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            real_A = test_transform(real_A)
            real_A = real_A.to(device)
            real_A = torch.unsqueeze(real_A, dim=0)

            fake_B = netG_A2B(real_A)
            AtoB = torch.cat([real_A, fake_B], dim=3)

            fake_B = reverse(opt, fake_B)
            AtoB = reverse(opt, AtoB)

            if index < 10:
                cv2.imwrite(os.path.join(AtoB_dir, '{}.png'.format(index+1)), AtoB)
            if index < args.test_size:
                cv2.imwrite(os.path.join(fakeB_dir, '{}.png'.format(index + 1)), fake_B)
            else:
                break

        for index, p in enumerate(testB_paths):
            real_B = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            real_B = test_transform(real_B)
            real_B = real_B.to(device)
            real_B = torch.unsqueeze(real_B, dim=0)

            fake_A = netG_B2A(real_B)
            BtoA = torch.cat([real_B, fake_A], dim=3)

            fake_A = reverse(opt, fake_A)
            BtoA = reverse(opt, BtoA)

            if index < 10:
                cv2.imwrite(os.path.join(BtoA_dir, '{}.png'.format(index+1)), BtoA)
            if index < args.test_size:
                cv2.imwrite(os.path.join(fakeA_dir, '{}.png'.format(index + 1)), fake_A)
            else:
                break


if __name__ == "__main__":
    Test_CycleGAN(args=opt)
