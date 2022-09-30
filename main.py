import argparse
from train import TrainCycleGAN

# Arguments
parser = argparse.ArgumentParser(description='Train CycleGAN')

parser.add_argument('--gpu_num', type=int, default=0)

# Training parameters
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--offset_epochs', type=int, default=0)
parser.add_argument('--decay_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--GAN_loss_alpha', type=float, default=1.0)
parser.add_argument('--cycle_loss_beta', type=float, default=10.0)
parser.add_argument('--identity_loss_gamma', type=float, default=5.0)

# Dataset
parser.add_argument('--train_size', type=int, default=3000)
parser.add_argument('--domain1', type=str, default='Dog')
parser.add_argument('--domain2', type=str, default='Cat')

# Transformations
parser.add_argument('--resize', type=bool, default=True)
parser.add_argument('--crop', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--flip', type=bool, default=True)
parser.add_argument('--normalize', type=bool, default=True)

args = parser.parse_args()

train_CycleGAN = TrainCycleGAN(args)
train_CycleGAN.train()
