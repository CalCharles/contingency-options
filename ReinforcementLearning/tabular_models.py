import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from ReinforcementLearning.models import Model

class TabularQ(Model):
    def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=None):
        super(TabularQ, self).__init__(args, num_inputs, num_outputs, name=name, factor=factor, minmax=minmax)
        self.Qtable = dict()
        self.action_prob_table = dict()
        self.initial_value = 1.0 # need to set this
        self.initial_aprob = 1 / num_outputs

    def forward(self, x):
        '''
        TODO: make use of time_estimator, link up Q vals and action probs
        TODO: clean up cuda = True to something that is actually true
        TODO: only accepts integer array input states of form (num in batch, num_vals)
        '''
        Qvals = []
        aprobs = []
        for xv in x: # for each x in the batch, convert state to hash and get Q value
            if len(xv.shape) > 1:
                xv = xv[0]
            hsh = tuple(int(v) for v in xv)
            if hsh in self.Qtable:
                Qval = self.Qtable[hsh]
                Aprob = self.action_prob_table[hsh]
            else:
                Aprob = torch.Tensor([self.initial_aprob for _ in range(self.num_outputs)]).cuda()
                Qval = torch.Tensor([self.initial_value for _ in range(self.num_outputs)]).cuda()
                self.Qtable[hsh] = Qval
                self.action_prob_table[hsh] = Aprob
            Qvals.append(Qval)
            aprobs.append(Aprob)
        Qvals = torch.stack(Qvals,dim=0)
        aprobs = torch.stack(aprobs,dim=0)
        action_probs = pytorch_model.wrap(aprobs, cuda=self.iscuda)
        Q_vals = pytorch_model.wrap(Qvals, cuda=self.iscuda)
        # print(Q_vals.shape, action_probs.shape)
        values = Q_vals.max(dim=1)[0]
        probs = F.softmax(action_probs, dim=1)
        # print("probs", probs)
        log_probs = F.log_softmax(action_probs, dim=1)

        dist_entropy = -(log_probs * probs).sum(-1).mean()

        return values, dist_entropy, probs, Q_vals

class TileCoding(Model):
    pass
