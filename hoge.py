import dgl
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
from dgl.data import CoraGraphDataset, CiteseerGraphDataset

dataset = CoraGraphDataset()
print(dataset)

print()

dataset = CiteseerGraphDataset()
print(dataset)