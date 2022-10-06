import argparse
from train import TrainCycleGAN

# Arguments
parser = argparse.ArgumentParser(description='Train CycleGAN')

parser.add_argument('--exp_detail', default='Finetune StyleGAN', type=str)
parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--seed', default=100, type=int)

# Training parameters
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--offset_epochs', type=int, default=0)
parser.add_argument('--decay_epochs', type=int, default=100)
parser.add_argument('--checkpoint_interval', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--GAN_loss_alpha', type=float, default=1.0)
parser.add_argument('--cycle_loss_beta', type=float, default=10.0)
parser.add_argument('--identity_loss_gamma', type=float, default=5.0)

# Dataset
parser.add_argument('--domain1', type=str, default='FFHQ')
parser.add_argument('--domain2', type=str, default='Dog')
parser.add_argument('--domain1_size', type=int, default=1000)
parser.add_argument('--domain2_size', type=int, default=1000)
parser.add_argument('--test_size', type=int, default=100)

# Transformations
parser.add_argument('--resize', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--flip', type=bool, default=True)
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--mean', type=tuple, default=(0.5, 0.5, 0.5))
parser.add_argument('--std', type=tuple, default=(0.5, 0.5, 0.5))

args = parser.parse_args()

train_CycleGAN = TrainCycleGAN(args)
train_CycleGAN.train()
