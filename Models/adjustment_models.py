import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from Models.models import Model, pytorch_model, models

class ConnectionModel(Model): # TODO: make ConnectionModel handle progressive-network style connections
    def __init__(self, args, model, connections):
        super().__init__(args=args, num_inputs=1, num_outputs=1, factor=8, needs_final=False) # only args used
        self.remove_last()
        self.model = model
        self.connections = connections # TODO: make modulelist
        self.layers = []
        if self.model is not None:
            self.layers.append(self.model)
        if self.connections is not None:
            self.layers.append (self.connections)
        self.init_form = args.init_form
        self.iscuda = args.cuda

    def forward(self, x, resp):
        if self.model is not None:
            x = self.model.hidden(x, resp)
        if self.connections is not None:
            x = self.connections(x)
        return x

class InputDefinedBias(Model):
    def __init__(self, args, num_inputs, hidden_size, indef=True):
        super().__init__(args=args, num_inputs=num_inputs, num_outputs=num_inputs, factor=8, needs_final=False)
        self.num_inputs = num_inputs
        self.indef = indef
        self.hidden_size = hidden_size
        self.layers = []
        if not self.indef:
            self.l1 = nn.Parameter(torch.tensor([0.0 for i in range(self.num_inputs)]))
            self.l1.requires_grad = True
        else:
            self.l1 = nn.Linear(num_inputs, self.hidden_size)
            self.outp = nn.Linear(self.hidden_size, num_inputs)
            self.layers.append(self.outp)
        self.layers.append(self.l1)

    def forward(self, x):
        if self.indef:
            y = self.l1(x)
            y = F.relu(y)
            y = self.outp(y)
        else:
            y = self.l1 * 3
        x += y
        # print(x.abs().mean(), y.abs().mean(), y)
        return self.acti(x)



class AdjustmentModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        args, num_inputs, num_outputs, factor = self.get_args(kwargs)
        self.args = args
        factor = int(args.factor)
        self.keep_initial_output = args.keep_initial_output
        # initial_model, base_model right now is a dummy, must call load_base
        if args.adjustment_form == 'none':
            self.initial_model = None
            model = None
        else:
            self.initial_model = models[args.adjustment_form](**kwargs)
            kwargs['needs_final'] = False
            model = models[args.adjustment_form](**kwargs)
        self.post = True
        self.correction_form = args.correction_form
        if args.correction_form == 'none':
            correct = None
            self.post = 1
        elif args.correction_form == 'linearpost':
            correct = nn.Linear(self.insize, self.insize)
            self.post = 1
        elif args.correction_form == 'linearsum':
            correct = nn.Linear(self.insize, self.insize)
            self.post = 2
        elif args.correction_form == 'inbiaspost':
            correct = InputDefinedBias(args, num_outputs, self.insize, indef=True)
            self.post = 1
        elif args.correction_form == 'inbiaspre':
            correct = InputDefinedBias(args, num_inputs, self.insize, indef=True)
            self.post = 0
        elif args.correction_form == 'biaspost':
            correct = InputDefinedBias(args, num_outputs, self.insize, indef=False)
            self.post = 1
        elif args.correction_form == 'biaspre':
            correct = InputDefinedBias(args, num_inputs, self.insize, indef=False)
            self.post = 0

        self.adjusted_model = ConnectionModel(args, model, correct)

        self.hidden_size = self.num_inputs*factor*factor // min(2,factor)
        print("Network Sizes: ", self.num_inputs, self.insize, self.hidden_size)
        self.freeze_initial = args.freeze_initial
        self.option_index = 0
        if self.keep_initial_output:
            self.remove_last()
            self.layers.append(self.adjusted_model)
        # self.l1 = nn.Linear(self.num_inputs, self.num_inputs*factor*factor)

    def hidden(self, x, resp):
        # print(x.shape)
        if self.post == 0:
            x = self.adjusted_model(x, resp) # TODO: unless x is the same dimension, invalidates resp
        base = self.initial_model.hidden(x, resp)
        if self.post == 1:
            base = self.adjusted_model(base, resp)
        if self.post == 2:
            x = self.adjusted_model(base, resp)
            return self.acti(x + base) # TODO: multiple ways to combine
        return base

    def forward(self, x, resp):
        '''
        TODO: make use of time_estimator, link up Q vals and action probs
        TODO: clean up cuda = True to something that is actually true
        '''
        x = self.preamble(x, resp)
        x = self.hidden(x, resp)
        if self.keep_initial_output:
            values, dist_entropy, probs, Q_vals = self.initial_model.last_layer(x)
        else:
            values, dist_entropy, probs, Q_vals = self.last_layer(x)
        return values, dist_entropy, probs, Q_vals


    def load_base(self, initial_model):
        self.initial_model = initial_model# sets the parameters of a model to the parameter values given as a single long vector
        if self.freeze_initial:
            self.initial_model.requires_grad = False
        if type(self.adjusted_model.connections) is nn.Linear:
            if self.adjusted_model.model is None:
                insize = self.insize
            else:
                insize = self.adjusted_model.model.insize
            self.adjusted_model.connections = nn.Linear(self.initial_model.insize, insize)
            self.adjusted_model.layers[-1] = self.adjusted_model.connections
        elif self.correction_form.find('post') != -1:
            self.adjusted_model.connections = InputDefinedBias(self.args, self.initial_model.insize, self.insize, indef=self.correction_form.find('in') != -1)
            self.adjusted_model.layers[-1] = self.adjusted_model.connections
        print("loading: ", initial_model.name, self)
        # print(self.adjusted_model.get_parameters(), self.adjusted_model.init_form)
        self.train()
        self.reset_parameters()
        # print(self.adjusted_model.get_parameters())
        # print(self.adjusted_model.connections)

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
            params = self.adjusted_model.set_parameters(param_val)
            return params
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

