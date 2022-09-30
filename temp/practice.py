import torch
import torch.nn as nn

input = torch.Tensor(1, 2, 10, 10)
print(input.data[0].shape)
