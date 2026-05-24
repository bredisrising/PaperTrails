import torch
import numpy as np
import matplotlib.pyplot as plt
import random

dim = 16

weight_matrix = torch.randn((dim, dim))

mean = torch.rand(1) * 100
std = torch.rand(1) * 6

input_vector = torch.randn((1, dim)) * std + mean



x = input_vector @ weight_matrix 

# layer norm
mean = x.mean()
std = x.std()
gain = torch.ones_like(x)
layernormed = (x - mean) / std  * gain

result = torch.where(x < 0, 0.0, x)





