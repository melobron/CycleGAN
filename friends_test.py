import argparse
import cv2
from model import Generator
from friends_dataset import FriendsDataset
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
parser.add_argument('--crop', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=512)
parser.add_argument('--flip', type=bool, default=False)
parser.add_argument('--normalize', type=bool, default=True)

args = parser.parse_args()


def get_transforms2(args):
    transform_list = [transforms.ToTensor()]
    if args.resize:
        transform_list.append(transforms.Resize(int(args.patch_size*1.12), transforms.InterpolationMode.BICUBIC))
    if args.crop:
        transform_list.append(transforms.CenterCrop(args.patch_size))
    if args.flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    if args.normalize:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transform_list


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

    transform = transforms.Compose(get_transforms2(args))
    test_dataset = FriendsDataset(transform=transform)

    save_dir = './results/friends_faces'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for index, face in enumerate(test_dataset):
        face = face.to(device)
        face = torch.unsqueeze(face, dim=0)

        fake_B = 0.5*(netG_A2B(face)+1.0)
        real_A = 0.5*(face+1.0)

        AtoB = torch.cat([real_A, fake_B], dim=3)
        AtoB = torch.squeeze(AtoB).cpu()
        AtoB = tensor_to_numpy(AtoB)
        AtoB = cv2.cvtColor(AtoB, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(save_dir, '{}.png'.format(index+1)), AtoB)


if __name__ == "__main__":
    Test_CycleGAN(args=args)
