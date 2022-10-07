import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import adaptive_avg_pool2d
from scipy import linalg

import argparse
import cv2
import numpy as np
from tqdm import tqdm

from inception import InceptionV3
from utils import *

# Arguments
parser = argparse.ArgumentParser(description='Evaluate FID')

parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--seed', type=int, default=100)

# Train
parser.add_argument('--batch_size', type=int, default=50)

# Dataset
parser.add_argument('--dir1', type=str, default='../datasets/FFHQ/test')
parser.add_argument('--dir2', type=str, default='./experiments/exp3/results/fake_FFHQ')
# parser.add_argument('--dir2', type=str, default='../datasets/Dog/test')

# InceptionV3 Network
""" Build Pretrained InceptionV3 (Input: 299 x 299 x 3)
1. Output Blocks: Choose blocks to return features of
- 0: output of 1st max pooling -> 64 x 73 x 73
- 1: output of 2nd max pooling -> 192 x 35 x 35
- 2: output which is fed to aux classifier -> 768 x 17 x 17
- 3: output of final average pooling -> 2048 x 1 x 1
2. Requires Grad: True when fine-tuning
3. Use FID Inception: Choose Implementation
- True: pretrained model used in Tensorflow's FID implementation
- False: pretrained model used in Torchvision's FID implementation
"""
parser.add_argument('--output_blocks', type=tuple, default=(3,))
parser.add_argument('--requires_grad', type=bool, default=False)
parser.add_argument('--use_fid_inception', type=bool, default=True)

opt = parser.parse_args()


class EvaluationDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_paths = make_dataset(img_dir)
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_numpy = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_numpy)
        return img_tensor

    def __len__(self):
        return len(self.img_paths)


def compute_activation_statistics(model, device, img_dir, transform, batch_size, output_block):
    model.eval().to(device)

    dataset = EvaluationDataset(img_dir=img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    dim_dict = {0: 64, 1: 192, 2: 768, 3: 2048}
    dims = dim_dict[output_block]
    acts = np.empty(shape=(len(dataset), dims))

    start_index = 0
    for index, imgs in enumerate(tqdm(dataloader)):
        imgs = imgs.to(device)

        with torch.no_grad():
            act = model(imgs)[0]

        # When dim is not 2048, pred require global spatial average pooling
        if act.shape[2] != 1 or act.shape[3] != 1:
            act = adaptive_avg_pool2d(act, output_size=(1, 1))

        act = act.squeeze(3).squeeze(2).cpu().numpy()
        acts[start_index: start_index + act.shape[0]] = act

        start_index += act.shape[0]

    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """ The Frechet Distance between 2 multivariate Gaussians
    d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)) """

    assert mu1.shape == mu2.shape, 'Mean Vectors have different shape'
    assert sigma1.shape == sigma2.shape, 'Std Vectors have different shape'

    diff = mu1 - mu2

    # Check if singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print('FID calculation produces singular product, adding {} to diagonal of cov estimates'.format(eps))
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def Evaluate_FID(args):
    # Device
    device = torch.device('cuda:{}'.format(opt.gpu_num))

    # Random Seeds
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)

    # Model
    model = InceptionV3(output_blocks=args.output_blocks, requires_grad=args.requires_grad, use_fid_inception=args.use_fid_inception)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(299, transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    mu1, sigma1 = compute_activation_statistics(model=model, device=device, img_dir=args.dir1, transform=transform,
                                                batch_size=args.batch_size, output_block=args.output_blocks[0])
    mu2, sigma2 = compute_activation_statistics(model=model, device=device, img_dir=args.dir2, transform=transform,
                                                batch_size=args.batch_size, output_block=args.output_blocks[0])

    fid_value = calculate_frechet_distance(mu1=mu1, sigma1=sigma1, mu2=mu2, sigma2=sigma2)

    return fid_value


if __name__ == "__main__":
    fid_value = Evaluate_FID(args=opt)
    print('FID: {}'.format(fid_value))
