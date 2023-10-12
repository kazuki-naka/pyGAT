import dgl
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
from dgl.data import CoraGraphDataset, CiteseerGraphDataset

import loralib as lora
import torch

# dataset = CoraGraphDataset()
# print(dataset)

# print()

# dataset = CiteseerGraphDataset()
# print(dataset)

in_features = 64
out_features = 128
batch_size = 13

layer = lora.Linear(in_features, out_features, r=16)

# `x`: (13, 64)
x = torch.rand(batch_size, in_features)
# `y`: (13, 128)
# (13, 64) * (64, 128) --> (13, 128)
y = layer(x)

print(f"x.shape: {x.shape}")
print(f"y.shape: {y.shape}")

# print(type(layer))
