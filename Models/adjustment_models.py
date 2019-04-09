import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from Models.models import Model, pytorch_model, models

class ConnectionModel(Model): # TODO: make ConnectionModel handle progressive-network style connections
    def __init__(self, args, model, connections):
        super().__init__(args, 1, 1) # only args used
        self.remove_last()
        self.model = model
        self.connections = connections # TODO: make modulelist
        self.layers = [self.connections]
        self.init_form = args.init_form
        self.iscuda = args.cuda

    def forward(self, x, resp):
        x = self.model.hidden(x, resp)
        x = self.connections(x)
        return x

    def reset_parameters(self):
        super().reset_parameters()
        self.model.reset_parameters()

class AdjustmentModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        args, num_inputs, num_outputs, factor = self.get_args(kwargs)
        factor = int(args.factor)
        # initial_model, base_model right now is a dummy, must call load_base
        self.initial_model = models[args.adjustment_form](**kwargs)
        model = models[args.adjustment_form](**kwargs)
        model.remove_last()
        correct = nn.Linear(self.insize, self.insize)
        self.adjusted_model = ConnectionModel(args, model, correct)
        self.hidden_size = self.num_inputs*factor*factor // min(2,factor)
        print("Network Sizes: ", self.num_inputs, self.insize, self.hidden_size)
        self.freeze_initial = args.freeze_initial
        self.layers = []
        self.option_index = 0
        # self.l1 = nn.Linear(self.num_inputs, self.num_inputs*factor*factor)

    def hidden(self, x, resp):
        # print(x.shape)
        base = self.initial_model.hidden(x, resp)
        x = self.adjusted_model(x, resp)
        return self.acti(x + base) # TODO: multiple ways to combine

    def load_base(self, initial_model):
        self.initial_model = initial_model# sets the parameters of a model to the parameter values given as a single long vector
        if self.freeze_initial:
            self.initial_model.requires_grad = False
        self.adjusted_model.connections = nn.Linear(self.adjusted_model.model.insize, self.initial_model.insize)
        print("loading: ", initial_model.name, self)
        self.train()
        self.reset_parameters()

    def get_parameters(self):
        if self.freeze_initial:
            return self.adjusted_model.get_parameters()
        else:
            return super().get_parameters()

    def count_parameters(self, reuse=False):
        if self.freeze_initial:
            self.parameter_count = self.adjusted_model.count_parameters(reuse=reuse)
        else:
            self.parameter_count = super().count_parameters(reuse=reuse)
        return self.parameter_count 

    def set_parameters(self, param_val):
        if self.freeze_initial:
            return self.adjusted_model.set_parameters(param_val)
        else:
            return super().set_parameters(param_val)

    def reset_parameters(self):
        super().reset_parameters()
        if not self.freeze_initial:
            self.initial_model.count_parameters(reuse=False)
        self.adjusted_model.reset_parameters()
        print ("resetting: ", self.parameter_count)

    # def forward(self, x, resp):
    #     '''
    #     TODO: make use of time_estimator, link up Q vals and action probs
    #     '''
    #     x = self.hidden(x, resp)
    #     values, dist_entropy, probs, Q_vals = self.initial_model.last_layer(x)
    #     return values, dist_entropy, probs, Q_vals

    def compute_layers(self, x):
        layer_outputs = self.initial_model.compute_layers(x) + self.base_model.compute_layers(x)
        return layer_outputs

