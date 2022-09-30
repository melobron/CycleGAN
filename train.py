import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import itertools
import time
import pandas as pd
import matplotlib.pyplot as plt

from model import Generator, Discriminator
from datasets import PairedDataset
from utils import *


class TrainCycleGAN:
    def __init__(self, args):
        # Arguments
        self.args = args

        # Device
        self.gpu_num = args.gpu_num
        self.device = torch.device('cuda:{}'.format(self.gpu_num) if torch.cuda.is_available() else 'cpu')

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
        self.batch_size = args.batch_size

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
        self.optimizer_D_A = optim.Adam(self.netD_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.optimizer_D_B = optim.Adam(self.netD_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

        # Scheduler
        self.scheduler_G = optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=LambdaLR(
            args.n_epochs, args.offset_epochs, args.decay_epochs).step)
        self.scheduler_D_A = optim.lr_scheduler.LambdaLR(self.optimizer_D_A, lr_lambda=LambdaLR(
            args.n_epochs, args.offset_epochs, args.decay_epochs).step)
        self.scheduler_D_B = optim.lr_scheduler.LambdaLR(self.optimizer_D_B, lr_lambda=LambdaLR(
            args.n_epochs, args.offset_epochs, args.decay_epochs).step)

        # Transform
        transform = transforms.Compose(get_transforms(args))

        # Dataset
        self.dataset = PairedDataset(domain1=args.domain1, domain2=args.domain2, train_size=args.train_size, train=True, transform=transform)
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True)

        # Replay Buffer
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        # Save Paths
        self.project_name = '{}2{}_train_size_{}'.format(args.domain1, args.domain2, args.train_size)
        self.checkpoint_path = './checkpoints/{}/'.format(self.project_name)
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        self.result_path = './results/{}/'.format(self.project_name)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

    def train(self):
        print(self.device)

        # Losses
        G_total_loss_list, G_identity_loss_list, G_GAN_loss_list, G_cycle_loss_list = [], [], [], []
        D_A_loss_list, D_B_loss_list, D_total_loss_list = [], [], []

        start = time.time()
        for epoch in range(1, self.n_epochs + 1):

            # Training
            for batch, data in enumerate(self.dataloader):
                real_A, real_B = data['domain1'], data['domain2']
                real_A, real_B = real_A.to(self.device), real_B.to(self.device)

                # Generators
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
                loss_G = self.alpha * (loss_GAN_A2B + loss_GAN_B2A) + \
                         self.beta * (loss_cycle_ABA + loss_cycle_BAB) + \
                         self.gamma * (loss_identity_A + loss_identity_B)

                loss_G.backward()
                self.optimizer_G.step()

                # Discriminator A
                self.optimizer_D_A.zero_grad()

                # Real Loss
                pred_real = self.netD_A(real_A)
                loss_D_real = self.criterion_GAN(pred_real, target_real)

                # Fake Loss
                fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                pred_fake = self.netD_A(fake_A.detach())
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
                fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                pred_fake = self.netD_B(fake_B.detach())
                loss_D_fake = self.criterion_GAN(pred_fake, target_fake)

                # Total Loss
                loss_D_B = (loss_D_real + loss_D_fake) / 2
                loss_D_B.backward()
                self.optimizer_D_B.step()

                # Save Losses
                G_total_loss_list.append(loss_G.item())
                G_identity_loss_list.append(loss_identity_A.item() + loss_identity_B.item())
                G_GAN_loss_list.append(loss_GAN_A2B.item() + loss_GAN_B2A.item())
                G_cycle_loss_list.append(loss_cycle_ABA.item() + loss_cycle_BAB.item())
                D_A_loss_list.append(loss_D_A.item())
                D_B_loss_list.append(loss_D_B.item())
                D_total_loss_list.append(loss_D_A.item() + loss_D_B.item())

                print(
                    '[Epoch {}][{}/{}] | loss: G_total={:.3f} G_identity={:.3f} G_GAN={:.3f} G_Cycle={:.3f} D_total={:.3f}'.format(
                        epoch, (batch + 1) * self.batch_size, len(self.dataset), loss_G.item(),
                        (loss_identity_A.item() + loss_identity_B.item()), (loss_GAN_A2B.item() + loss_GAN_B2A.item()),
                        (loss_cycle_ABA.item() + loss_cycle_BAB.item()), (loss_D_A.item() + loss_D_B.item())
                    ))

            self.scheduler_G.step()
            self.scheduler_D_A.step()
            self.scheduler_D_B.step()

            # Checkpoints
            if epoch % 50 == 0 or epoch == self.n_epochs:
                torch.save(self.netG_A2B.state_dict(), os.path.join(self.checkpoint_path,
                                                                    'netG_A2B_{}epochs.pth'.format(epoch)))
                torch.save(self.netG_B2A.state_dict(), os.path.join(self.checkpoint_path,
                                                                    'netG_B2A_{}epochs.pth'.format(epoch)))
                torch.save(self.netD_A.state_dict(), os.path.join(self.checkpoint_path,
                                                                  'netD_A_{}epochs.pth'.format(epoch)))
                torch.save(self.netD_B.state_dict(), os.path.join(self.checkpoint_path,
                                                                  'netD_B_{}epochs.pth'.format(epoch)))

        # Visualize Loss
        df_G_total = pd.DataFrame(G_total_loss_list)
        df_G_identity = pd.DataFrame(G_identity_loss_list)
        df_G_GAN = pd.DataFrame(G_GAN_loss_list)
        df_G_cycle = pd.DataFrame(G_cycle_loss_list)
        df_D_A = pd.DataFrame(D_A_loss_list)
        df_D_B = pd.DataFrame(D_B_loss_list)
        df_D_total = pd.DataFrame(D_total_loss_list)

        fig, axs = plt.subplots(4, 2)
        axs[0, 0].plot(df_G_identity)
        axs[0, 0].set_title('G Identity Loss')
        axs[0, 1].plot(df_G_GAN)
        axs[0, 1].set_title('G GAN Loss')
        axs[1, 0].plot(df_G_cycle)
        axs[1, 0].set_title('G Cycle Loss')
        axs[1, 1].plot(df_G_total)
        axs[1, 1].set_title('G Total Loss')
        axs[2, 0].plot(df_D_A)
        axs[2, 0].set_title('D A Loss')
        axs[2, 1].plot(df_D_B)
        axs[2, 1].set_title('D B Loss')
        axs[3, 0].plot(df_D_total)
        axs[3, 0].set_title('D Total Loss')

        plt.tight_layout()
        plt.savefig(os.path.join(self.result_path, 'evaluation.png'))
        plt.show()
