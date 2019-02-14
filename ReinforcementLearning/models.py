import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


class pytorch_model():
    def __init__(self, combiner=None, loss=None, reducer=None, cuda=True):
        # should have customizable combiner and loss, but I dont.
        self.cuda=cuda
        self.reduce_size = 2 # someday this won't be hard coded either

    @staticmethod
    def wrap(data, dtype=torch.float, cuda=True):
        # print(Variable(torch.Tensor(data).cuda()))
        if cuda:
            return Variable(torch.tensor(data, dtype=dtype).cuda())
        else:
            return Variable(torch.tensor(data, dtype=dtype))

    @staticmethod
    def unwrap(data):
        return data.data.cpu().numpy()

    @staticmethod
    def concat(data, axis=0):
        return torch.cat(data, dim=axis)

class Model(nn.Module):
    def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=None):
        super(Model, self).__init__()
        num_inputs = int(num_inputs)
        num_outputs = int(num_outputs)
        self.insize = max(num_inputs * num_outputs * factor * factor, num_inputs * num_outputs)
        self.minmax = minmax
        if minmax is not None:
            self.minmax = (torch.cat([pytorch_model.wrap(minmax[0]).cuda() for _ in range(args.num_stack)], dim=0), torch.cat([pytorch_model.wrap(minmax[1]).cuda() for _ in range(args.num_stack)], dim=0))
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.layers = []
        self.critic_linear = nn.Linear(self.insize, 1)
        self.time_estimator = nn.Linear(self.insize, 1)
        self.QFunction = nn.Linear(self.insize, num_outputs)
        self.action_probs = nn.Linear(self.insize, num_outputs)
        self.name = name
        self.iscuda = args.cuda # TODO: don't just set this to true
        Model.reset_parameters(self)

    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('relu')
        # nn.init.uniform_(self.critic_linear.weight.data, .9 / self.insize, 1.1 / self.insize)
        # nn.init.uniform_(self.time_estimator.weight.data, .9 / self.insize, 1.1 / self.insize)
        # nn.init.uniform_(self.QFunction.weight.data, .9 / self.insize, 1.1 / self.insize)
        # nn.init.uniform_(self.action_probs.weight.data, .9 / self.insize, 1.1 / self.insize)
        nn.init.uniform_(self.critic_linear.weight.data, 0.0, 1.1 / self.insize)
        nn.init.uniform_(self.time_estimator.weight.data, 0.0, 1.1 / self.insize)
        nn.init.uniform_(self.QFunction.weight.data, 0.0, 1.1 / self.insize)
        nn.init.uniform_(self.action_probs.weight.data, 0.0, 1.1 / self.insize)
        nn.init.uniform_(self.critic_linear.bias.data, 0.0,.1)
        nn.init.uniform_(self.time_estimator.bias.data, 0.0,.1)
        nn.init.uniform_(self.QFunction.bias.data, 0.0,.1)
        nn.init.uniform_(self.action_probs.bias.data, 0.0,.1)

        # nn.init.uniform_(self.critic_linear.bias.data, -.1,.1)
        # nn.init.uniform_(self.time_estimator.bias.data, -.1,.1)
        # nn.init.uniform_(self.QFunction.bias.data, -.1,.1)
        # nn.init.uniform_(self.action_probs.bias.data, -.1,.1)

    def normalize(self, x):
        return (x - self.minmax[0]) / (self.minmax[1] - self.minmax[0])

    def forward(self, x):
        '''
        TODO: make use of time_estimator?, link up Q vals and action probs
        TODO: clean up cuda = True to something that is actually true
        input: [batch size, state size] (TODO: no multiple processes)
        output [batch size, 1], [batch size, 1], [batch_size, num_actions], [batch_size, num_actions]
        '''
        if self.minmax is None:
            pass
        else:
            x = self.normalize(x)
        action_probs = self.action_probs(x)
        Q_vals = self.QFunction(x)
        values = self.critic_linear(x)
        probs = F.softmax(action_probs, dim=1)
        log_probs = F.log_softmax(action_probs, dim=1)

        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return values, dist_entropy, probs, Q_vals

    def save(self, pth):
        torch.save(self, pth + self.name + ".pt")

class BasicModel(Model):
    def __init__(self, args, num_inputs, num_outputs, name="option", minmax=None):
        super(BasicModel, self).__init__(args, num_inputs, num_outputs, name=name, factor=args.factor, minmax=minmax)
        factor = int(args.factor)
        self.hidden_size = self.num_inputs*factor*factor
        print("Network Sizes: ", self.num_inputs, self.num_inputs*factor*factor, self.insize)
        # self.l1 = nn.Linear(self.num_inputs, self.num_inputs*factor*factor)
        self.l1 = nn.Linear(self.num_inputs,self.insize)

        self.l2 = nn.Linear(self.num_inputs*factor*factor, self.insize)
        self.layers.append(self.l1)
        self.layers.append(self.l2)
        super().reset_parameters()
        self.train()
        self.reset_parameters()


    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('relu')
        # nn.init.uniform_(self.l1.weight.data, .9 / self.num_inputs, 1.1 / self.num_inputs)
        # nn.init.uniform_(self.l2.weight.data, .9 / self.hidden_size, 1.1 / self.hidden_size)
        nn.init.uniform_(self.l1.weight.data, 0.0, 1.1 / self.num_inputs / 10.0)
        nn.init.uniform_(self.l2.weight.data, 0.0, 1.1 / self.hidden_size / 10.0)
        nn.init.uniform_(self.l1.bias.data, 0.0,.1)
        nn.init.uniform_(self.l2.bias.data, 0.0,.1)

    def forward(self, x):
        '''
        TODO: make use of time_estimator, link up Q vals and action probs
        TODO: clean up cuda = True to something that is actually true
        '''
        # print(x.shape)
        if self.minmax is None:
            pass
        else:
            x = self.normalize(x)
        x = self.l1(x)
        x = F.relu(x)
        # x = self.l2(x)
        # x = F.relu(x)
        # print(self.l2.weight)
        # print(x.shape)
        action_probs = self.action_probs(x)
        Q_vals = self.QFunction(x)
        # print(self.action_probs.weight)
        values = self.critic_linear(x)
        probs = F.softmax(action_probs, dim=1)
        log_probs = F.log_softmax(action_probs, dim=1)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        # print("lp, p", action_probs, log_probs, probs)
        # print(values.shape, probs.shape, dist_entropy.shape, Q_vals.shape)

        return values, dist_entropy, probs, Q_vals

from ReinforcementLearning.basis_models import FourierBasisPolicy, GaussianBasisPolicy
from ReinforcementLearning.tabular_models import TabularQ, TileCoding

models = {"basic": BasicModel, "tab": TabularQ, "tile": TileCoding, "fourier": FourierBasisPolicy, "gaussian": GaussianBasisPolicy}
