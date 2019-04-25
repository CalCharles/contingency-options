import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import copy, os
from file_management import default_value_arg

class pytorch_model():
    def __init__(self, combiner=None, loss=None, reducer=None, cuda=True):
        # should have customizable combiner and loss, but I dont.
        self.cuda=cuda
        self.reduce_size = 2 # someday this won't be hard coded either

    @staticmethod
    def wrap(data, dtype=torch.float, cuda=True):
        # print(Variable(torch.Tensor(data).cuda()))
        if type(data) == torch.Tensor:
            return data.clone().detach() 
        else:
            if cuda:
                return Variable(torch.tensor(data, dtype=dtype).cuda())
            else:
                return Variable(torch.tensor(data, dtype=dtype))

    @staticmethod
    def unwrap(data):
        return data.clone().detach().cpu().numpy()

    @staticmethod
    def concat(data, axis=0):
        return torch.cat(data, dim=axis)



class Model(nn.Module):
    def __init__(self, **kwargs):
        super(Model, self).__init__()
        args, num_inputs, num_outputs, factor = self.get_args(kwargs)
        self.minmax = default_value_arg(kwargs, 'minmax', None)
        self.name = default_value_arg(kwargs, 'name', 'option')
        self.no_preamble = default_value_arg(kwargs, 'no_preamble', False)
        self.param_dim = default_value_arg(kwargs, 'param_dim', 1)
        self.option_values = torch.zeros(1, self.param_dim) # changed externally to the parameters
        self.parameterized_option = 0
        num_inputs = int(num_inputs)
        num_outputs = int(num_outputs)
        self.num_layers = args.num_layers
        if args.num_layers == 0:
            self.insize = num_inputs
        else:
            self.insize = factor * factor * factor // min(factor, 8)
        print("MINMAX", self.minmax)
        if self.minmax is not None:
            self.minmax = (torch.cat([pytorch_model.wrap(self.minmax[0] - 1e-5).cuda() for _ in range(args.num_stack)], dim=0), torch.cat([pytorch_model.wrap(self.minmax[1] + 1e-5).cuda() for _ in range(args.num_stack)], dim=0))
            for mm in self.minmax:
                mm.requires_grad = False
        print("MINMAX", self.minmax)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        if args.model_form in ["gaussian", "fourier", "gaumulti", "gaudist"]:
            self.insize = args.factor # should get replaced in basis function section
        self.critic_linear = nn.Linear(self.insize, 1)
        self.time_estimator = nn.Linear(self.insize, 1)
        self.QFunction = nn.Linear(self.insize, num_outputs)
        self.action_probs = nn.Linear(self.insize, num_outputs)
        self.layers = [self.critic_linear, self.time_estimator, self.QFunction, self.action_probs]
        self.iscuda = args.cuda # TODO: don't just set this to true
        self.use_normalize = args.normalize
        self.init_form = args.init_form 
        self.scale = args.scale
        if args.activation == "relu":
                self.acti = F.relu
        elif args.activation == "sin":
            self.acti = torch.sin
        elif args.activation == "sigmoid":
            self.acti = F.sigmoid
        elif args.activation == "tanh":
            self.acti = F.tanh
        self.test = not args.train # testing mode for evaluation
            
    def get_args(self, kwargs):
        return kwargs['args'], kwargs['num_inputs'], kwargs['num_outputs'], kwargs['factor']

    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('relu')
        for layer in self.layers:
            if type(layer) == nn.Conv2d:
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            else:
                if self.init_form == "uni":
                    # print("div", layer.weight.data.shape[0], layer.weight.data.shape)
                    nn.init.uniform_(layer.weight.data, 0.0, 3 / layer.weight.data.shape[0])
                if self.init_form == "smalluni":
                    # print("div", layer.weight.data.shape[0], layer.weight.data.shape)
                    nn.init.uniform_(layer.weight.data, 0.0, 1 / layer.weight.data.shape[0])
                elif self.init_form == "xnorm":
                    torch.nn.init.xavier_normal_(layer.weight.data)
                elif self.init_form == "xuni":
                    torch.nn.init.xavier_uniform_(layer.weight.data)
                elif self.init_form == "eye":
                    torch.nn.init.eye_(layer.weight.data)
                if layer.bias is not None:                
                    nn.init.uniform_(layer.bias.data, 0.0, 1e-6)
        print("parameter number", self.count_parameters(reuse=False))

    def preamble(self, x, resp):
        # if self.no_preamble:
        if self.minmax is not None and self.use_normalize:
            x = self.normalize(x)
        x = x * self.scale
        return x

        # nn.init.uniform_(self.critic_linear.weight.data, .9 / self.insize, 1.1 / self.insize)
        # nn.init.uniform_(self.time_estimator.weight.data, .9 / self.insize, 1.1 / self.insize)
        # nn.init.uniform_(self.QFunction.weight.data, .9 / self.insize, 1.1 / self.insize)
        # nn.init.uniform_(self.action_probs.weight.data, .9 / self.insize, 1.1 / self.insize)
        # nn.init.uniform_(self.critic_linear.weight.data, 0.0, 1.1 / self.insize)
        # nn.init.uniform_(self.time_estimator.weight.data, 0.0, 1.1 / self.insize)
        # nn.init.uniform_(self.QFunction.weight.data, 0.0, 1.1 / self.insize)
        # nn.init.uniform_(self.action_probs.weight.data, 0.0, 1.1 / self.insize)
        # nn.init.uniform_(self.critic_linear.bias.data, 0.0,.1)
        # nn.init.uniform_(self.time_estimator.bias.data, 0.0,.1)
        # nn.init.uniform_(self.QFunction.bias.data, 0.0,.1)
        # nn.init.uniform_(self.action_probs.bias.data, 0.0,.1)

        # nn.init.uniform_(self.critic_linear.bias.data, -.1,.1)
        # nn.init.uniform_(self.time_estimator.bias.data, -.1,.1)
        # nn.init.uniform_(self.QFunction.bias.data, -.1,.1)
        # nn.init.uniform_(self.action_probs.bias.data, -.1,.1)

    def normalize(self, x):
        return (x - self.minmax[0]) / (abs(self.minmax[1] - self.minmax[0]) + 1e-10)

    def last_layer(self, x):
        '''
        TODO: make use of time_estimator?, link up Q vals and action probs
        TODO: clean up cuda = True to something that is actually true
        input: [batch size, state size] (TODO: no multiple processes)
        output [batch size, 1], [batch size, 1], [batch_size, num_actions], [batch_size, num_actions]
        '''
        action_probs = self.action_probs(x)
        Q_vals = self.QFunction(x)
        values = self.critic_linear(x)
        probs = F.softmax(action_probs, dim=1)
        # print(probs)
        log_probs = F.log_softmax(action_probs, dim=1)

        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return values, dist_entropy, probs, Q_vals

    def forward(self, x, resp):
        '''
        TODO: make use of time_estimator, link up Q vals and action probs
        TODO: clean up cuda = True to something that is actually true
        '''
        x = self.preamble(x, resp)
        x = self.hidden(x, resp)
        values, dist_entropy, probs, Q_vals = self.last_layer(x)
        return values, dist_entropy, probs, Q_vals

    def save(self, pth):
        torch.save(self, os.path.join(pth, self.name + ".pt"))

    def get_parameters(self):
        params = []
        for param in self.parameters():
            params.append(param.data.flatten())
        return torch.cat(params)

    def get_gradients(self):
        grads = []
        for param in self.parameters():
            grads.append(param.grad.data.flatten())
        return torch.cat(grads)

    def set_parameters(self, param_val): # sets the parameters of a model to the parameter values given as a single long vector
        if len(param_val) != self.count_parameters():
            raise ValueError('invalid number of parameters to set')
        pval_idx = 0
        for param in self.parameters():
            param_size = np.prod(param.size())
            cur_param_val = param_val[pval_idx : pval_idx+param_size]
            param.data = torch.from_numpy(cur_param_val) \
                              .reshape(param.size()).float()
            pval_idx += param_size
        if self.iscuda:
            self.cuda()

    # count number of parameters
    def count_parameters(self, reuse=True):
        if reuse and self.parameter_count > 0:
            return self.parameter_count
        self.parameter_count = 0
        for param in self.parameters():
            # print(param.size(), np.prod(param.size()), self.insize, self.hidden_size)
            self.parameter_count += np.prod(param.size())
        return self.parameter_count

    # TODO: write code to remove last layer if unnecessary
    def remove_last(self):
        self.critic_linear = None
        self.time_estimator = None
        self.QFunction = None
        self.action_probs = None
        self.layers = self.layers[4:]


class BasicModel(Model):    
    def __init__(self, **kwargs):
        super(BasicModel, self).__init__(**kwargs)
        args, num_inputs, num_outputs, factor = self.get_args(kwargs)
        self.hidden_size = factor*factor*factor // min(4,factor)
        print("Network Sizes: ", self.num_inputs, self.insize, self.hidden_size)
        # self.l1 = nn.Linear(self.num_inputs, self.num_inputs*factor*factor)
        if args.num_layers == 1:
            self.l1 = nn.Linear(self.num_inputs,self.insize)
        elif args.num_layers == 2:
            self.l1 = nn.Linear(self.num_inputs,self.hidden_size)
            self.l2 = nn.Linear(self.hidden_size, self.insize)
        elif args.num_layers == 3:
            self.l1 = nn.Linear(self.num_inputs,self.hidden_size)
            self.l2 = nn.Linear(self.hidden_size,self.hidden_size)
            self.l3 = nn.Linear(self.hidden_size, self.insize)
        if args.num_layers > 0:
            self.layers.append(self.l1)
        if args.num_layers > 1:
            self.layers.append(self.l2)
        if args.num_layers > 2:
            self.layers.append(self.l3)
        self.train()
        self.reset_parameters()

    def hidden(self, x, resp):
        # print(x.shape)
        # if self.minmax is not None and self.use_normalize:
        #     x = self.normalize(x)
        # x = x * self.scale
        # if torch.isnan(x.sum()):
        #     print("in", x)
        # inp = x

        # print("normin", x)
        if self.num_layers > 0:
            x = self.l1(x)
            x = self.acti(x)
        # if torch.isnan(x.sum()):
        #     print("l1", x, inp)
        if self.num_layers > 1:
            x = self.l2(x)
            x = self.acti(x)
        # if torch.isnan(x.sum()):
        #     print("l2", x, inp)
        if self.num_layers > 2:
            x = self.l3(x)
            x = self.acti(x)
        # if torch.isnan(x.sum()):
        #     print("l3", x, inp)
        # if np.random.rand() < .001:
        #     print("total", x.sum().detach(), self.l1.weight.abs().sum().detach(), self.action_probs.weight.abs().sum().detach())
        return x

    def compute_layers(self, x):
        layer_outputs = []
        if self.minmax is not None and self.use_normalize:
            x = self.normalize(x)
        if self.num_layers > 0:
            x = self.l1(x)
            x = F.relu(x)
            layer_outputs.append(x.clone())
        if self.num_layers > 1:
            x = self.l2(x)
            x = F.relu(x)
            layer_outputs.append(x.clone())

        return layer_outputs

class DistributionalModel(BasicModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        args, num_inputs, num_outputs, factor = self.get_args(kwargs)
        self.value_bounds = args.value_bounds
        self.num_value_atoms = args.num_value_atoms
        self.dz = (self.value_bounds[1] - self.value_bounds[0]) / (self.num_value_atoms - 1)
        self.value_support = pytorch_model.wrap([self.value_bounds[0] + (i * self.dz) for i in range(self.num_value_atoms)], cuda = args.cuda)
        self.value_support.requires_grad = False
        self.value_distribution = nn.Linear(self.insize, args.num_value_atoms * num_outputs)
        self.layers.append(value_distribution)

    def forward(self, x):
        # if self.minmax is not None and self.use_normalize:
        #     x = self.normalize(x)
        # print("normin", x)
        if self.num_layers > 0:
            x = self.l1(x)
            x = F.relu(x)
        if self.num_layers > 1:
            x = self.l2(x)
            x = F.relu(x)
        Q_vals = self.compute_Qval(x)
        values = Q_vals.max(dim=1)[0]
        action_probs = self.action_probs(x)
        probs = F.softmax(action_probs, dim=1)
        log_probs = F.log_softmax(action_probs, dim=1)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return values, dist_entropy, probs, Q_vals

    def compute_Qval(self, x):
        # a = (a * torch.ones(x.shape[0])).long()
        x = self.compute_value_distribution_mid(x)
        return (self.value_support * x).sum(dim=2)
        
    def compute_value_distribution_mid(self, x):
        '''
        states as [batch size, final layer size]
        actions as [batch, 1]
        '''
        # a = torch.zeros((actions.shape[0], self.num_outputs))
        # if self.iscuda:
        #     a = a.cuda()
        # a[list(range(actions.shape[0])), actions.squeeze()] = 1.0
        # x = torch.cat((x,a), dim=1)
        x = self.value_distribution(x)
        x = x.view(-1, self.num_outputs, self.num_value_atoms)
        probs = F.softmax(x, dim=2)
        return probs

    def compute_value_distribution(self, x):
        '''
        states as [batch size, state size]
        actions as [batch, 1]
        '''
        if self.minmax is not None and self.use_normalize:
            x = self.normalize(x)
        # print("normin", x)
        if self.num_layers > 0:
            x = self.l1(x)
            x = F.relu(x)
        if self.num_layers > 1:
            x = self.l2(x)
            x = F.relu(x)
        return self.compute_value_distribution_mid(x)


from Models.basis_models import FourierBasisModel, GaussianBasisModel, GaussianMultilayerModel, GaussianDistributionModel
from Models.tabular_models import TabularQ, TileCoding
from Models.image_models import ObjectSumImageModel
models = {"basic": BasicModel, "dist": DistributionalModel, "gaudist": GaussianDistributionModel, "tab": TabularQ, 
            "tile": TileCoding, "fourier": FourierBasisModel, "gaussian": GaussianBasisModel, "gaumulti": GaussianMultilayerModel,
            "sumimage": ObjectSumImageModel}
from Models.parameterized_models import ParameterizedOneHotModel, ParameterizedContinuousModel, ParameterizedBoostDim
models["paramhot"] = ParameterizedOneHotModel
models["paramcont"] = ParameterizedContinuousModel
models["paramboost"] = ParameterizedBoostDim
from Models.adjustment_models import AdjustmentModel
models["adjust"] = AdjustmentModel
from Models.multi_models import PopulationModel
models["population"] = PopulationModel
