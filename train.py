import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import itertools
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

from model import Generator, Discriminator
from datasets import TranslationDataset
from utils import *


class TrainCycleGAN:
    def __init__(self, args):
        # Arguments
        self.args = args

        # Device
        self.gpu_num = args.gpu_num
        self.device = torch.device('cuda:{}'.format(self.gpu_num) if torch.cuda.is_available() else 'cpu')

        # Random Seeds
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        # Models
        self.netG_A2B = Generator().to(self.device)
        self.netG_B2A = Generator().to(self.device)
        self.netD_A = Discriminator().to(self.device)
        self.netD_B = Discriminator().to(self.device)

        self.netG_A2B.apply(weights_init_normal)
        self.netG_B2A.apply(weights_init_normal)
        self.netD_A.apply(weights_init_normal)
        self.netD_B.apply(weights_init_normal)

        # Training Parameters
        self.n_epochs = args.n_epochs
        self.offset_epochs = args.offset_epochs
        self.decay_epochs = args.decay_epochs
        self.checkpoint_interval = args.checkpoint_interval

        self.batch_size = args.batch_size
        self.lr = args.lr
        self.mean = args.mean
        self.std = args.std

        self.domain1 = args.domain1
        self.domain2 = args.domain2
        self.domain1_size = args.domain1_size
        self.domain2_size = args.domain2_size
        self.test_size = args.test_size

        self.alpha = args.GAN_loss_alpha
        self.beta = args.cycle_loss_beta
        self.gamma = args.identity_loss_gamma

        # Loss
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()

        # Optimizer
        self.optimizer_G = optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
                                      lr=args.lr, betas=(0.5, 0.999))
        self.optimizer_D_A = optim.Adam(self.netD_A.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D_B = optim.Adam(self.netD_B.parameters(), lr=self.lr, betas=(0.5, 0.999))

        # Scheduler
        self.scheduler_G = optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=LambdaLR(
            args.n_epochs, args.offset_epochs, args.decay_epochs).step)
        self.scheduler_D_A = optim.lr_scheduler.LambdaLR(self.optimizer_D_A, lr_lambda=LambdaLR(
            args.n_epochs, args.offset_epochs, args.decay_epochs).step)
        self.scheduler_D_B = optim.lr_scheduler.LambdaLR(self.optimizer_D_B, lr_lambda=LambdaLR(
            args.n_epochs, args.offset_epochs, args.decay_epochs).step)

        # Transform
        train_transform = transforms.Compose(get_transforms(args))
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        # Dataset
        self.train_dataset = TranslationDataset(domain1=self.domain1, domain2=self.domain2,
                                                domain1_size=self.domain1_size, domain2_size=self.domain2_size,
                                                train=True, transform=train_transform)
        # self.test_dataset = TranslationDataset(domain1=self.domain1, domain2=self.domain2,
        #                                        domain1_size=self.test_size, domain2_size=self.test_size,
        #                                        train=False, transform=test_transform)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)

        # Replay Buffer
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        # Directories
        self.exp_dir = make_exp_dir('./experiments/')['new_dir']
        self.exp_num = make_exp_dir('./experiments/')['new_dir_num']
        self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.result_path = os.path.join(self.exp_dir, 'results')

        # Tensorboard
        self.summary = SummaryWriter('runs/exp{}'.format(self.exp_num))

    def prepare(self):
        # Save Paths
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        # Save Argument file
        param_file = os.path.join(self.exp_dir, 'params.json')
        with open(param_file, mode='w') as f:
            json.dump(self.args.__dict__, f, indent=4)

    def train(self):
        print(self.device)
        self.prepare()

        for epoch in range(1, self.n_epochs + 1):
            with tqdm(self.train_dataloader, desc='Epoch {}'.format(epoch)) as tepoch:
                ########### Training ###########
                for batch, data in enumerate(tepoch):
                    real_A, real_B = data['domain1'], data['domain2']
                    real_A, real_B = real_A.to(self.device), real_B.to(self.device)

                    ########### Update D ###########
                    self.optimizer_G.zero_grad()

                    # GAN Loss
                    fake_B = self.netG_A2B(real_A)
                    pred_fake = self.netD_B(fake_B)
                    target_real = torch.ones_like(pred_fake, requires_grad=False).to(self.device)
                    loss_GAN_A2B = self.criterion_GAN(pred_fake, target_real)

                    fake_A = self.netG_B2A(real_B)
                    pred_fake = self.netD_B(fake_A)
                    loss_GAN_B2A = self.criterion_GAN(pred_fake, target_real)

                    # Cycle Loss
                    reconstruction_A = self.netG_B2A(fake_B)
                    loss_cycle_ABA = self.criterion_cycle(reconstruction_A, real_A)

                    reconstruction_B = self.netG_A2B(fake_A)
                    loss_cycle_BAB = self.criterion_cycle(reconstruction_B, real_B)

                    # Identity Loss
                    same_A = self.netG_B2A(real_A)
                    loss_identity_A = self.criterion_identity(same_A, real_A)

                    same_B = self.netG_A2B(real_B)
                    loss_identity_B = self.criterion_identity(same_B, real_B)

                    # Total Loss
                    loss_G = self.alpha * (loss_GAN_A2B + loss_GAN_B2A)/2 + \
                             self.beta * (loss_cycle_ABA + loss_cycle_BAB)/2 + \
                             self.gamma * (loss_identity_A + loss_identity_B)/2

                    loss_G.backward()
                    self.optimizer_G.step()

                    ########### Update D ###########
                    # Discriminator A
                    self.optimizer_D_A.zero_grad()

                    # Real Loss
                    pred_real = self.netD_A(real_A)
                    loss_D_real = self.criterion_GAN(pred_real, target_real)

                    # Fake Loss
                    fake_A_ = self.fake_A_buffer.push_and_pop(fake_A)
                    pred_fake = self.netD_A(fake_A_.detach())
                    target_fake = torch.zeros_like(pred_fake, requires_grad=False).to(self.device)
                    loss_D_fake = self.criterion_GAN(pred_fake, target_fake)

                    # Total Loss
                    loss_D_A = (loss_D_real + loss_D_fake) / 2
                    loss_D_A.backward()
                    self.optimizer_D_A.step()

                    # Discriminator B
                    self.optimizer_D_B.zero_grad()

                    # Real Loss
                    pred_real = self.netD_B(real_B)
                    loss_D_real = self.criterion_GAN(pred_real, target_real)

                    # Fake Loss
                    fake_B_ = self.fake_B_buffer.push_and_pop(fake_B)
                    pred_fake = self.netD_B(fake_B_.detach())
                    loss_D_fake = self.criterion_GAN(pred_fake, target_fake)

                    # Total Loss
                    loss_D_B = (loss_D_real + loss_D_fake) / 2
                    loss_D_B.backward()
                    self.optimizer_D_B.step()

                    ########### Save Results ###########
                    tepoch.set_postfix(G_loss=loss_G.item(), DA_loss=loss_D_A.item(), DB_loss=loss_D_B.item())
                    self.summary.add_scalar('G_loss', loss_G.item(), epoch)
                    self.summary.add_scalar('DA_loss', loss_D_A.item(), epoch)
                    self.summary.add_scalar('DB_loss', loss_D_B.item(), epoch)

                ########### Validation ###########



            self.scheduler_G.step()
            self.scheduler_D_A.step()
            self.scheduler_D_B.step()

            # Checkpoints
            if epoch % self.checkpoint_interval == 0 or epoch == self.n_epochs:
                torch.save(self.netG_A2B.state_dict(), os.path.join(self.checkpoint_dir, 'netG_A2B_{}epochs.pth'.format(epoch)))
                torch.save(self.netG_B2A.state_dict(), os.path.join(self.checkpoint_dir, 'netG_B2A_{}epochs.pth'.format(epoch)))

