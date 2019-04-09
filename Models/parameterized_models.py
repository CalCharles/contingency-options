import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from Models.models import Model, pytorch_model, models
from file_management import default_value_arg

class ParametrizedModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        args, num_inputs, num_outputs, factor = self.get_args(kwargs)
        factor = int(args.factor)
        # initial_model, base_model right now is a dummy, must call load_base
        # Input size should be augmented with parameter size
        add_size = self.get_option_dim(kwargs['param_dim'])
        if 'class_sizes' in kwargs:
            kwargs['class_sizes'].append(add_size) # param_dim must be included
        kwargs['num_inputs'] += add_size
        self.model = models[args.parameterized_form](**kwargs)
        self.layers = []
        self.option_index = 0
        # self.l1 = nn.Linear(self.num_inputs, self.num_inputs*factor*factor)
        self.train()
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

    # def forward(self, x, resp):
    #     '''
    #     TODO: make use of time_estimator, link up Q vals and action probs
    #     '''
    #     x = self.hidden(x)
    #     values, dist_entropy, probs, Q_vals = super().forward(x)
    #     return values, dist_entropy, probs, Q_vals

    def compute_layers(self, x):
        return self.model.compute_layers(x)


class ParameterizedOneHotModel(ParametrizedModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_options = default_value_arg(kwargs, 'param_dim', 0) # number of inputs for the options
        self.hot_encodings = []
        self.option_index = 0

    def get_option_dim(self, param_dim):
        return param_dim

    def create_option_vec(self, batch_size):
        vec = torch.zeros(batch_size, self.num_options)
        vec[:, self.option_index] = 1.0
        if self.iscuda:
            vec = vec.cuda()
        return vec
    

class ParameterizedContinuousModel(ParametrizedModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hot_encodings = []
        self.option_dim = default_value_arg(kwargs, 'param_dim', 0)
        self.option_value = torch.zeros(param_dim) # changed externally based on option
    
    def get_option_dim(self, param_dim):
        return param_dim

    def create_option_vec(self, batch_size):
        vec = torch.stack([self.option_value.clone() for _ in range(batch_size)])
        if self.iscuda:
            vec = vec.cuda()
        return vec