import argparse
import cv2
from model import Generator
from datasets import PairedDataset
from utils import *

# Arguments
parser = argparse.ArgumentParser(description='Test CycleGAN')

parser.add_argument('--gpu_num', type=int, default=0)

# Training parameters
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=1)

# Dataset
parser.add_argument('--domain1', type=str, default='RealandFake_faces')
parser.add_argument('--domain2', type=str, default='Dog')
parser.add_argument('--train_size', type=int, default=1000)

# Transformations
parser.add_argument('--resize', type=bool, default=True)
parser.add_argument('--crop', type=bool, default=False)
parser.add_argument('--patch_size', type=int, default=229)
parser.add_argument('--flip', type=bool, default=False)
parser.add_argument('--normalize', type=bool, default=True)

args = parser.parse_args()


def Test_CycleGAN(args):
    device = torch.device('cuda:{}'.format(args.gpu_num))

    netG_A2B = Generator().to(device)
    netG_B2A = Generator().to(device)

    netG_A2B_name = 'netG_A2B_{}epochs.pth'.format(args.n_epochs)
    netG_B2A_name = 'netG_B2A_{}epochs.pth'.format(args.n_epochs)

    dataset_name = '{}2{}_train_size_{}'.format(args.domain1, args.domain2, args.train_size)
    netG_A2B.load_state_dict(torch.load('./checkpoints/{}/{}'.format(dataset_name, netG_A2B_name), map_location=device))
    netG_B2A.load_state_dict(torch.load('./checkpoints/{}/{}'.format(dataset_name, netG_B2A_name), map_location=device))

    netG_A2B.eval()
    netG_B2A.eval()

    transform = transforms.Compose(get_transforms(args))
    test_dataset = PairedDataset(domain1=args.domain1, domain2=args.domain2, train_size=args.train_size, train=False, transform=transform)

    save_dir = './results/{}/{}epochs'.format(dataset_name, args.n_epochs)
    fake_domain1_dir = os.path.join(save_dir, '{}'.format(args.domain1))
    fake_domain2_dir = os.path.join(save_dir, '{}'.format(args.domain2))
    if not os.path.exists(fake_domain1_dir):
        os.makedirs(fake_domain1_dir)
    if not os.path.exists(fake_domain2_dir):
        os.makedirs(fake_domain2_dir)

    for index, data in enumerate(test_dataset):
        if index >= 30:
            break

        real_A, real_B = data['domain1'], data['domain2']
        real_A, real_B = real_A.to(device), real_B.to(device)
        real_A, real_B = torch.unsqueeze(real_A, dim=0), torch.unsqueeze(real_B, dim=0)

        fake_B = 0.5*(netG_A2B(real_A)+1.0)
        fake_A = 0.5*(netG_B2A(real_B)+1.0)
        real_A = 0.5*(real_A+1.0)
        real_B = 0.5*(real_B+1.0)

        AtoB = torch.cat([real_A, fake_B], dim=3)
        BtoA = torch.cat([real_B, fake_A], dim=3)

        AtoB, BtoA = torch.squeeze(AtoB).cpu(), torch.squeeze(BtoA).cpu()
        AtoB, BtoA = tensor_to_numpy(AtoB), tensor_to_numpy(BtoA)
        AtoB, BtoA = cv2.cvtColor(AtoB, cv2.COLOR_RGB2BGR), cv2.cvtColor(BtoA, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(fake_domain1_dir, '{}.png'.format(index+1)), BtoA)
        cv2.imwrite(os.path.join(fake_domain2_dir, '{}.png'.format(index+1)), AtoB)


if __name__ == "__main__":
    Test_CycleGAN(args=args)
