import torch

from models.resnet.resnet50 import Resnet1

model = Resnet1(50, out_indices=(0, 1, 2, 3))
print(model)