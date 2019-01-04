import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from rl_template import pytorch_model

class Model(nn.Module):
    def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=(-1,-1)):
        super(Model, self).__init__()
        num_inputs = int(num_inputs)
        num_outputs = int(num_outputs)
        self.minmax = minmax
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.layers = []
        self.critic_linear = nn.Linear(max(num_inputs * num_outputs * factor, num_inputs * num_outputs), 1)
        self.time_estimator = nn.Linear(max(num_inputs*num_outputs * factor, num_inputs * num_outputs), 1)
        self.QFunction = nn.Linear(max(num_inputs*num_outputs * factor, num_inputs * num_outputs), num_outputs)
        self.action_probs = nn.Linear(max(num_inputs*num_outputs * factor, num_inputs * num_outputs), num_outputs)
        self.name = name
        self.cuda = args.cuda # TODO: don't just set this to true

    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('relu')
        nn.init.uniform_(self.critic_linear.weight.data, .9, 1.0)
        nn.init.uniform_(self.time_estimator.weight.data, .9, 1.0)
        nn.init.uniform_(self.QFunction.weight.data, .9, 1.0)
        nn.init.uniform_(self.action_probs.weight.data, .9, 1.0)
        nn.init.uniform_(self.critic_linear.bias.data, -.1,.1)
        nn.init.uniform_(self.time_estimator.bias.data, -.1,.1)
        nn.init.uniform_(self.QFunction.bias.data, -.1,.1)
        nn.init.uniform_(self.action_probs.bias.data, -.1,.1)

    def normalize(self, x):
        return (x - self.minmax[0]) / (self.minmax[1] - self.minmax[0])

    def forward(self, x):
        '''
        TODO: make use of time_estimator, link up Q vals and action probs
        TODO: clean up cuda = True to something that is actually true
        '''
        if self.minmax[0] == self.minmax[1] == -1:
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

class BasicModel(nn.Module):
    def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=(-1,-1)):
        super(BasicModel, self).__init__(args, num_inputs, num_outputs, name="option", factor=8, minmax=(-1,-1))
        self.l1 = nn.Linear(num_inputs, num_inputs*factor)
        self.l2 = nn.Linear(num_inputs*factor, max(num_inputs*num_outputs * factor // 4, num_inputs * num_outputs))
        self.layers.append(self.l1)
        self.layers.append(self.l2)
        super().reset_parameters()
        self.reset_parameters()

    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('relu')
        nn.init.uniform_(self.l1.weight.data, .9, 1.0)
        nn.init.uniform_(self.l2.weight.data, .9, 1.0)
        nn.init.uniform_(self.l1.bias.data, -.1,.1)
        nn.init.uniform_(self.l2.bias.data, -.1,.1)

    def normalize(self, x):
        return (x - self.minmax[0]) / (self.minmax[1] - self.minmax[0])

    def forward(self, x):
        '''
        TODO: make use of time_estimator, link up Q vals and action probs
        TODO: clean up cuda = True to something that is actually true
        '''
        if self.minmax[0] == self.minmax[1] == -1:
            pass
        else:
            x = self.normalize(x)
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)

        action_probs = self.action_probs(x)
        Q_vals = self.QFunction(x)
        values = self.critic_linear(x)
        probs = F.softmax(action_probs, dim=1)
        log_probs = F.log_softmax(action_probs, dim=1)

        dist_entropy = -(log_probs * probs).sum(-1).mean()

        return values, dist_entropy, probs, Q_vals

class TabularQ(nn.Module):
    def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=(-1,-1)):
        super(TabularQ, self).__init__(args, num_inputs, num_outputs, name="option", factor=8, minmax=(-1,-1))
        self.Qtable = dict()
        self.action_prob_table = dict()
        self.initial_value = 10 # need to set this
        self.initial_aprob = 1 / num_outputs

    def normalize(self, x):
        return (x - self.minmax[0]) / (self.minmax[1] - self.minmax[0])

    def forward(self, x):
        '''
        TODO: make use of time_estimator, link up Q vals and action probs
        TODO: clean up cuda = True to something that is actually true
        TODO: only accepts integer array input states of form (num in batch, 1, num_vals)
        '''
        Qvals = []
        aprobs = []
        for xv in x:
            hsh = tuple(int(v) for v in xv.squeeze())
            if hsh in self.table:
                Qval = self.Qtable[hsh]
                Aprob = self.action_prob_table[hsh]
            else:
                Aprob = [self.initial_aprob for _ in range(self.num_outputs)]
                Qval = [self.initial_value for _ in range(self.num_outputs)]
                self.Qtable[hsh] = Qval
                self.action_prob_table[hsh] = Aprob
            Qvals.append(Qval)
            aprobs.append(Aprob)

        action_probs = pytorch_model.wrap(Aprob, cuda=self.cuda).unsqueeze(1)
        Q_vals = pytorch_model.wrap(Qvals, cuda=self.cuda).unsqueeze(1)
        values = Qvals.max(dim=2)
        probs = F.softmax(action_probs, dim=1)
        log_probs = F.log_softmax(action_probs, dim=1)

        dist_entropy = -(log_probs * probs).sum(-1).mean()

        return values, dist_entropy, probs, Q_vals

