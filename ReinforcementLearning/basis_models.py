import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from ReinforcementLearning.models import Model


class FourierBasisPolicy(Model):
    def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=None):
        super(FourierOptionPolicy, self).__init__(args, num_inputs, num_outputs, name=name, factor=factor, minmax=minmax)
        '''
        factor is the order
        layers defines the variate (1 = univariate, 2 = paired, 3=all)
        object_extractors[0] is the current object
        the remainder are any correlate objects, with relative computations 
        computations are relative to pre_extracted state (just getters)
        '''
        self.num_stack = args.num_stack
        self.dim = num_inputs // self.hist_num
        self.order = factor + 1 # include zero order
        self.variate = args.num_layers % 10 # decide the relatinship
        self.layering = args.num_layers // 10 # defines different kinds of relationships
        self.layers= []
        self.period = args.eps
        self.base_vals = [i for i in range(self.order+1)]

        self.order_vector = []
        for i in range (self.order):
            self.order_vector.append(np.pi*2*i/self.period)
        self.order_vector = pytorch_model.wrap(np.array(self.order_vector))
        print("VARIATE", self.variate)
        if self.variate == 1:
            self.basis_matrix = pytorch_model.wrap(np.identity(self.num_inputs * (self.order)))
            self.basis_size = (self.order) * self.num_inputs
            self.QFunction = nn.Linear((self.order) * self.num_inputs, self.num_outputs, bias=False)
        elif num_layers == 2:
            # TODO: use HIST LEN = num stack to get groupings
            # Basis function are correlated across 1: time ONLY, 2: across each timestep independently, 3: across both, but not simultaniously 
            id_matrix = np.identity(self.num_inputs * (self.order)) # defines the independent basis
            base_indexes = []
            if self.layering == 1 or self.layering == 3:
                basis_matrix1 = np.zeros(((self.num_inputs) * (self.order), self.dim * (self.order) ** self.num_stack))
                for ipt in range(self.dim):
                    for i in range((self.order) ** (self.num_stack)):
                        bases = self.num_to_base(i, self.order, self.num_stack)
                        bases.reverse()
                        bases = [b + ((self.order) * ipt) + ((self.order) * self.dim * i) for i, b in enumerate(bases)]
                        bases_indexes.append(bases)
                    indexes = np.array([[i + (ipt * (self.order) ** self.num_stack) for _ in range(self.num_inputs)] for i in range((self.order) ** self.num_stack)]).flatten()
                    basis_matrix1 = basis_matrix1.T
                    basis_matrix1[indexes, np.array(bases_indexes).flatten()] = 1
                    basis_matrix1 = basis_matrix1.T
                    basis_size = self.dim * (self.order) ** self.num_stack
                self.QFunction = nn.Linear((self.order) ** (self.num_inputs), self.num_outputs, bias=False)
                basis_final = np.concatenate((basis_matrix1, id_matrix), dim = 0)
            elif self.layering == 2 or self.layering == 3:
                basis_matrix2 = np.zeros(((self.num_inputs) * (self.order), self.stack * (self.order) ** self.dim))
                for stk in range(self.stack):
                    for i in range((self.order) ** (self.dim)):
                        bases = self.num_to_base(i, self.order, self.dim)
                        bases.reverse()
                        bases = [b + ((self.order) * self.dim * stk) + ((self.order)* i) for i, b in enumerate(bases)]
                        bases_indexes.append(bases)
                    indexes = np.array([[i + (ipt * (self.order) ** self.dim) for _ in range(self.num_inputs)] for i in range((self.order) ** self.dim)]).flatten()
                    basis_matrix2 = basis_matrix2.T
                    basis_matrix2[indexes, np.array(bases_indexes).flatten()] = 1
                    basis_matrix2 = basis_matrix2.T
                    basis_size = self.dim * (self.order) ** self.num_stack
                self.QFunction = nn.Linear((self.order) ** (self.num_inputs), self.num_outputs, bias=False)
                self.basis_matrix = np.concatenate((basis_matrix2, id_matrix), dim = 0)
            elif self.layering == 3:
                self.basis_matrix = np.concatenate((basis_matrix1, basis_matrix2, id_matrix), dim = 0)
            print( "not implemented yet")
        elif num_layers == 3:
            # print((self.order), self.num_inputs)
            # print((self.order) ** (self.num_inputs))
            self.basis_matrix = np.zeros(((self.num_inputs) * (self.order), (self.order) ** (self.num_inputs)))
            bases_indexes = []
            for i in range((self.order) ** (self.num_inputs)):
                bases = self.num_to_base(i, self.order, self.num_inputes) # every possible enumeration of indexes, which has (self.order) ** (self.num_inputs)
                bases.reverse()
                bases = [b + (self.order) * i for i, b in enumerate(bases)] #the indexes that we want to have activated, which is one per order
                bases_indexes.append(bases)
            # print(self.basis_matrix.shape)
            indexes = np.array([[i for _ in range(self.num_inputs)] for i in range((self.order) ** (self.num_inputs))]).flatten()
            self.basis_matrix = self.basis_matrix.T
            self.basis_matrix[indexes, np.array(bases_indexes).flatten()] = 1
            self.basis_matrix = self.basis_matrix.T
            self.basis_matrix = pytorch_model.wrap(self.basis_matrix)
            self.basis_size = (self.order) ** (self.num_inputs)
            self.QFunction = nn.Linear((self.order) ** (self.num_inputs), self.num_outputs, bias=False)

        # print("order vector", self.order_vector)
        # self.time_estimator = nn.Linear(max(num_inputs*num_outputs * factor, num_inputs * num_outputs), 1)
        self.action_probs = nn.Linear(self.num_outputs, self.num_outputs) # TODO: action propbs are based on Q values

        self.train()
        self.reset_parameters()
        # print("done initializing")

    def num_to_base(self, val, base, num_digits):
        num = []
        for _ in range(num_inputs):
            # print(val % base, val, base)
            num.append(val % base)
            val = val // base
        # num.reverse() # currently least to greatest
        return num

    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('relu')
        nn.init.uniform_(self.action_probs.weight.data, .9 / self.num_outputs * 2, 1.1 / self.num_outputs * 2)
        nn.init.uniform_(self.QFunction.weight.data, .9 / self.basis_size, 1.1 / self.basis_size)
        nn.init.uniform_(self.action_probs.bias.data, -.1,.1)

    def fourier_basis(self, inputs):
        # print("minmax", self.min_, self.max_)
        if self.minmax is not None:
            inputs = self.normalize(inputs)
        # for loops are supposed to be bad practice, but we will just keep these for now
        bat = []
        # print("ord", self.order_vector)
        for datapt in inputs:
            basis = []
            for val in datapt:
                # print ("input single", val)
                basis.append(torch.cos(val * self.order_vector))
            # print(basis)
            basis = torch.cat(basis)
            bat.append(basis)
        return torch.stack(bat)

    def forward(self, inputs):
        x = self.fourier_basis(inputs) # TODO: dimension
        # print(self.basis_matrix.shape, x.shape)
        # print("xprebasis", x, self.basis_matrix)
        x = torch.mm(x.view(-1,x.shape[1]), self.basis_matrix)
        # print("xbasis", x)
        Qvals = self.QFunction(x.view(x.shape[0],-1))
        aprobs = self.action_probs(Qvals)
        values = Qvals.max(dim=1)[0]
        probs = F.softmax(aprobs, dim=1)
        # print("probs", probs)
        log_probs = F.log_softmax(aprobs, dim=1)

        dist_entropy = -(log_probs * probs).sum(-1).mean()

        # print("xval", x)
        return values, dist_entropy, aprobs, Qvals

class GaussianBasisPolicy(Model):
    def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=None):
        super(GaussianBasisPolicy, self).__init__(args, num_inputs, num_outputs, name=name, factor=factor, minmax=minmax)
        '''
        factor is the order
        layers defines the variate (1 = univariate, 2 = paired, 3=all)
        object_extractors[0] is the current object
        the remainder are any correlate objects, with relative computations 
        computations are relative to pre_extracted state (just getters)
        '''
        self.hist_num = args.num_stack
        self.order = factor
        self.variate = args.num_layers
        self.layers= []
        self.period = args.eps
        self.base_vals = [i for i in range(self.order+1)]

        self.order_vector = []
        for i in range (self.order):
            self.order_vector.append(np.pi*2*i/self.period)
        self.order_vector = pytorch_model.wrap(np.array(self.order_vector))
        print("VARIATE", self.variate)
        if self.variate == 1:
            self.basis_matrix = pytorch_model.wrap(np.identity(self.num_inputs * (self.order)))
            self.basis_size = (self.order) * self.num_inputs
            self.QFunction = nn.Linear((self.order) * self.num_inputs, self.num_outputs, bias=False)
        elif num_layers == 2:
            # TODO: use HIST LEN = num stack to get groupings
            print( "not implemented yet")
        elif num_layers == 3:
            # print((self.order), self.num_inputs)
            # print((self.order) ** (self.num_inputs))
            self.basis_matrix = np.zeros(((self.num_inputs) * (self.order), (self.order) ** (self.num_inputs)))
            bases_indexes = []
            for i in range((self.order) ** (self.num_inputs)):
                bases = self.num_to_base(i)
                bases = [b + (self.order) * i for i, b in enumerate(bases)]
                bases_indexes.append(bases)
            # print(self.basis_matrix.shape)
            indexes = np.array([[i for _ in range(self.num_inputs)] for i in range((self.order) ** (self.num_inputs))]).flatten()
            self.basis_matrix = self.basis_matrix.T
            self.basis_matrix[indexes, np.array(bases_indexes).flatten()] = 1
            self.basis_matrix = self.basis_matrix.T
            self.basis_matrix = pytorch_model.wrap(self.basis_matrix)
            self.basis_size = (self.order) ** (self.num_inputs)
            self.QFunction = nn.Linear((self.order) ** (self.num_inputs), self.num_outputs, bias=False)

        # print("order vector", self.order_vector)
        # self.time_estimator = nn.Linear(max(num_inputs*num_outputs * factor, num_inputs * num_outputs), 1)
        self.action_probs = nn.Linear(self.num_outputs, self.num_outputs) # TODO: action propbs are based on Q values

        self.train()
        self.reset_parameters()
        # print("done initializing")

    def num_to_base(self, val):
        num = []
        base = self.order
        for _ in range(self.num_inputs):
            # print(val % base, val, base)
            num.append(val % base)
            val = val // base
        # num.reverse() # currently least to greatest
        return num

    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('relu')
        nn.init.uniform_(self.action_probs.weight.data, .9 / self.num_outputs * 2, 1.1 / self.num_outputs * 2)
        nn.init.uniform_(self.QFunction.weight.data, .9 / self.basis_size, 1.1 / self.basis_size)
        nn.init.uniform_(self.action_probs.bias.data, -.1,.1)

        print("after reset", self.QFunction.weight.data)

    def fourier_basis(self, inputs):
        # print("minmax", self.min_, self.max_)
        if self.minmax is not None:
            inputs = self.normalize(inputs)
        # for loops are supposed to be bad practice, but we will just keep these for now
        bat = []
        # print("ord", self.order_vector)
        for datapt in inputs:
            basis = []
            for val in datapt:
                # print ("input single", val)
                basis.append(torch.cos(val * self.order_vector))
            # print(basis)
            basis = torch.cat(basis)
            bat.append(basis)
        return torch.stack(bat)

    def forward(self, inputs):
        x = self.fourier_basis(inputs) # TODO: dimension
        # print(self.basis_matrix.shape, x.shape)
        # print("xprebasis", x, self.basis_matrix)
        x = torch.mm(x.view(-1,x.shape[1]), self.basis_matrix)
        # print("xbasis", x)
        Qvals = self.QFunction(x.view(x.shape[0],-1))
        aprobs = self.action_probs(Qvals)
        values = Qvals.max(dim=1)[0]
        probs = F.softmax(aprobs, dim=1)
        # print("probs", probs)
        log_probs = F.log_softmax(aprobs, dim=1)

        dist_entropy = -(log_probs * probs).sum(-1).mean()

        # print("xval", x)
        return values, dist_entropy, aprobs, Qvals
