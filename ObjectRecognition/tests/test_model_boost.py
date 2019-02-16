import numpy as np
import torch
from ChangepointDetection.LinearCPD import LinearCPD
from ObjectRecognition.model import ModelFocusBoost, ModelFocusCNN


net_params_1 = {
    'filter': 2,
    'channel': [3, 1],
    'kernel_size': [3, 5],
    'stride': [1, 1],
    'padding': [2, 4],
    'activation_fn': ['ReLU', 'Tanh']
}
net_params_2 = {
    'filter': 2,
    'channel': [10, 1],
    'kernel_size': [3, 5],
    'stride': [1, 1],
    'padding': [2, 4],
    'activation_fn': ['ReLU6', 'Tanh']
}
model_1 = ModelFocusCNN((84, 84), net_params=net_params_1)
model_2 = ModelFocusCNN((84, 84), net_params=net_params_2)
model_boost = ModelFocusBoost(
    LinearCPD(np.pi/4),
    model_1,
    model_2,
    train_flags=[True, False],
)

# parameters
print(model_boost.count_parameters())
ones = np.arange(model_boost.count_parameters())
model_boost.set_parameters(ones)
print(list(model_boost.parameters()))

# forward
out = model_boost.forward(torch.rand([100, 1, 84, 84]))
print(out, out.shape)
