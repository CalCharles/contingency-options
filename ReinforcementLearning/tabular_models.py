import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from ReinforcementLearning.models import Model, pytorch_model

class TabularModel(Model):
    def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=None):
        super().__init__(args, num_inputs, num_outputs, name=name, factor=factor, minmax=minmax)
        self.Qtable = dict()
        self.action_prob_table = dict()
        self.initial_value = 1.0 # need to set this
        self.initial_aprob = 1 / num_outputs

    def hash_function(self, xv):
        pass

    def get_Qval(self, hsh):
        pass

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
            hsh = self.hash_function(xv)
            Qval, Aprob = self.get_Qval(hsh)
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


class TabularQ(TabularModel):
    def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=None):
        super(TabularQ, self).__init__(args, num_inputs, num_outputs, name=name, factor=factor, minmax=minmax)
        self.Qtable = dict()
        self.action_prob_table = dict()
        self.initial_value = 1.0 # need to set this
        self.initial_aprob = 1 / num_outputs

    def get_Qval(self, hsh):
        if hsh in self.Qtable:
            Qval = self.Qtable[hsh]
            Aprob = self.action_prob_table[hsh]
        else:
            Aprob = torch.Tensor([self.initial_aprob for _ in range(self.num_outputs)]).cuda()
            Qval = torch.Tensor([self.initial_value for _ in range(self.num_outputs)]).cuda()
            self.Qtable[hsh] = Qval
            self.action_prob_table[hsh] = Aprob
        return Qval, Aprob


    def hash_function(self, xv):
        return tuple(int(v) for v in xv)

class TileCoding(TabularModel):
    def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=None):
        super(TileCoding, self).__init__(args, num_inputs, num_outputs, name=name, factor=factor, minmax=minmax)
        self.factor = args.factor
        minvs, maxvs = self.minmax
        self.tile_vectors = []
        for minv, maxv in zip(minvs, maxvs):
            order_vector = []
            for i in range (self.factor):
                print(minv + i * (maxv - minv) / (self.factor - 1))
                order_vector.append(float(minv + i * (maxv - minv) / (self.factor - 1)))
            self.tile_vectors.append(pytorch_model.wrap(np.array(order_vector)))

        self.Qtable = dict()
        self.action_prob_table = dict()
        self.initial_value = 1.0 # need to set this
        self.initial_aprob = 1 / num_outputs

    def get_Qval(self, hsh):
        if hsh in self.Qtable:
            Qval = self.Qtable[hsh]
            Aprob = self.action_prob_table[hsh]
        else:
            Aprob = torch.Tensor([self.initial_aprob for _ in range(self.num_outputs)]).cuda()
            Qval = torch.Tensor([self.initial_value for _ in range(self.num_outputs)]).cuda()
            self.Qtable[hsh] = Qval
            self.action_prob_table[hsh] = Aprob
        return Qval, Aprob


    def hash_function(self, xv):
        return tuple(int(torch.argmin((tv - v).abs())) for tv, v in zip(self.tile_vectors, xv))
