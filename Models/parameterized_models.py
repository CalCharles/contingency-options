import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from Models.models import Model, pytorch_model, models

class ParametrizedModel(Model):
    def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=None, sess=None, param_dim=-1):
        super().__init__(args, num_inputs, num_outputs, name=name, factor=factor, minmax=minmax, sess=None, param_dim=param_dim)
        factor = int(args.factor)
        # initial_model, base_model right now is a dummy, must call load_base
        # Input size should be augmented with parameter size
        self.model = models[args.parameterized_form](args, num_inputs, num_outputs, name=name, factor=factor, minmax=minmax, sess=sess, param_dim=param_dim)
        self.layers = []
        self.option_index = 0
        # self.l1 = nn.Linear(self.num_inputs, self.num_inputs*factor*factor)
        self.reset_parameters()

    def update_inputs(self, x):
        vec = self.create_option_vec(x.size(0)) # option vector creation differs
        x = torch.cat((x,vec), dim=1)
        return x

    def hidden(self, x):
        x = self.model.hidden(self.update_inputs(x))
        return x 

    def reset_parameters(self):
        super().reset_parameters()

    def forward(self, x):
        '''
        TODO: make use of time_estimator, link up Q vals and action probs
        '''
        x = self.hidden(x)
        values, dist_entropy, probs, Q_vals = super().forward(x)
        return values, dist_entropy, probs, Q_vals

    def compute_layers(self, x):
        return self.model.compute_layers(x)


class ParameterizedOneHotModel(ParametrizedModel):
    def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=None, sess=None, param_dim=-1):
        super().__init__(args, num_inputs + param_dim, num_outputs, name=name, factor=factor, minmax=minmax, sess=None, param_dim=param_dim)
        self.num_options = param_dim # number of inputs for the options
        self.hot_encodings = []
        self.option_index = 0

    def create_option_vec(self, batch_size):
        vec = torch.zeros(batch_size, self.num_options)
        vec[:, self.option_index] = 1.0
        if self.iscuda:
            vec = vec.cuda()
        return vec
    

class ParameterizedContinuousModel(ParametrizedModel):
    def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=None, sess=None, param_dim=-1):
        super().__init__(args, num_inputs + param_dim, num_outputs, name=name, factor=factor, minmax=minmax, sess=None, param_dim=param_dim)
        self.hot_encodings = []
        self.option_dim = param_dim
        self.option_value = torch.zeros(param_dim)

    def create_option_vec(self, batch_size):
        vec = torch.stack([self.option_value.clone() for _ in range(batch_size)])
        if self.iscuda:
            vec = vec.cuda()
        return vec