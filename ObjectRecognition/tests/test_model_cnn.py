import numpy as np
import torch
from ObjectRecognition.model import ModelFocusCNN


net_params = {
    'filter': 2,
    'channel': [3, 1],
    'kernel_size': [3, 5],
    'stride': [1, 1],
    'padding': [2, 4],
    'activation_fn': ['ReLU', 'Tanh']
}
model = ModelFocusCNN((84, 84), net_params=net_params)

# forward
out = model.forward(torch.zeros([100, 1, 84, 84]))
print(out, out.shape)

# parameters
print(model.count_parameters())
ones = np.arange(model.count_parameters())
model.set_parameters(ones)
print(list(model.parameters()))