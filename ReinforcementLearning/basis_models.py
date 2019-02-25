import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from ReinforcementLearning.models import Model, pytorch_model

class BasisModel(Model):
    def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=None, sess = None):
        ''' 
        initializes parameters and basis relations. 1 is fully independent (factor+1 * num_inputs) number of bases
        2 implies either time correlated (12), or state correlated(22), or both (02) (using tens place) (if input dim 4, around 40 is max order)
        3 implies fully correlated (factor+1 ^ num_inputs)
        '''
        super().__init__(args, num_inputs, num_outputs, name=name, factor=factor, minmax=minmax, sess=None)
        self.num_stack = args.num_stack
        self.dim = num_inputs // self.num_stack
        self.order = factor + 1 # include zero order
        self.variate = args.num_layers % 10 # decide the relatinship
        self.layering = args.num_layers // 10 # defines different kinds of relationships
        self.period = args.period
        self.base_vals = [i for i in range(self.order+1)]

        if self.variate == 1:
            self.basis_matrix = pytorch_model.wrap(np.identity(self.num_inputs * (self.order)))
            self.basis_size = (self.order) * self.num_inputs
            self.QFunction = nn.Linear((self.order) * self.num_inputs, self.num_outputs, bias=False)
            print(self.basis_size)
        elif self.variate == 2:
            # TODO: use HIST LEN = num stack to get groupings
            # Basis function are correlated across 1: time ONLY, 2: across each timestep independently, 3: across both, but not simultaniously 
            id_matrix = np.identity(self.num_inputs * (self.order)) # defines the independent basis
            if (self.layering == 1 or self.layering == 0) and self.dim > 1:
                basis_matrix1 = np.zeros(((self.num_inputs) * (self.order), self.dim * (self.order) ** self.num_stack))
                for ipt in range(self.dim):
                    basis_indexes = []
                    for i in range((self.order) ** (self.num_stack)):
                        bases = self.num_to_base(i, self.order, self.num_stack)
                        bases.reverse()
                        bases = [b + ((self.order) * ipt) + ((self.order) * self.dim * i) for i, b in enumerate(bases)]
                        # print(bases)
                        basis_indexes.append(bases)
                    # print(self.dim * (self.order) ** self.num_stack)
                    indexes = np.array([[i + (ipt * (self.order) ** self.num_stack) for _ in range(self.num_stack)] for i in range((self.order) ** self.num_stack)]).flatten()
                    # print(indexes)
                    basis_matrix1 = basis_matrix1.T
                    basis_matrix1[indexes, np.array(basis_indexes).flatten()] = 1
                    basis_matrix1 = basis_matrix1.T
                basis_size1 = self.dim * (self.order) ** self.num_stack
                self.basis_size = basis_size1 + (self.order * self.num_inputs)
                self.QFunction = nn.Linear((self.order) ** (self.num_inputs), self.num_outputs, bias=False)
                self.basis_matrix = pytorch_model.wrap(np.concatenate((basis_matrix1, id_matrix), axis = 1), cuda=args.cuda)
                # print(self.basis_matrix.shape)
            if (self.layering == 2 or self.layering == 0) and self.num_stack > 1:
                basis_matrix2 = np.zeros(((self.num_inputs) * (self.order), self.num_stack * (self.order) ** self.dim))
                for stk in range(self.num_stack):
                    basis_indexes = []
                    for i in range((self.order) ** (self.dim)):
                        bases = self.num_to_base(i, self.order, self.dim)
                        bases.reverse()
                        bases = [b + ((self.order) * self.dim * stk) + ((self.order)* i) for i, b in enumerate(bases)]
                        basis_indexes.append(bases)
                    indexes = np.array([[i + (stk * (self.order) ** self.dim) for _ in range(self.dim)] for i in range((self.order) ** self.dim)]).flatten()
                    # print(indexes)
                    basis_matrix2 = basis_matrix2.T
                    basis_matrix2[indexes, np.array(basis_indexes).flatten()] = 1
                    basis_matrix2 = basis_matrix2.T
                basis_size2 = self.num_stack * (self.order) ** self.dim
                self.basis_size = basis_size2 + (self.order * self.num_inputs)
                self.basis_matrix = pytorch_model.wrap(np.concatenate((basis_matrix2, id_matrix), axis = 1), cuda=args.cuda)
                # print(self.basis_matrix.shape)
                # np.set_printoptions(threshold=np.nan)
                # print("bm", basis_matrix2.T)
            if self.layering == 0:
                self.basis_size = basis_size1 + basis_size2 + (self.order * self.num_inputs)
                self.basis_matrix = pytorch_model.wrap(np.concatenate((basis_matrix1, basis_matrix2, id_matrix), axis = 1), cuda=args.cuda)
                # print(self.basis_matrix.shape)
        elif self.variate == 3:
            # print((self.order), self.num_inputs)
            # print((self.order) ** (self.num_inputs))
            self.basis_matrix = np.zeros(((self.num_inputs) * (self.order), (self.order) ** (self.num_inputs)))
            bases_indexes = []
            for i in range((self.order) ** (self.num_inputs)):
                bases = self.num_to_base(i, self.order, self.num_inputs) # every possible enumeration of indexes, which has (self.order) ** (self.num_inputs)
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
        print(self.basis_matrix.shape)
        self.QFunction = nn.Linear(self.basis_size, self.num_outputs, bias=False)
        self.action_probs = nn.Linear(self.basis_size, self.num_outputs, bias=False)
        self.layers[2] = self.QFunction
        self.layers[3] = self.action_probs
        self.reset_parameters()
        print(self.layers)
        self.basis_matrix.requires_grad = False

    def num_to_base(self, val, base, num_digits):
        num = []
        for _ in range(num_digits):
            # print(val % base, val, base)
            num.append(val % base)
            val = val // base
        # num.reverse() # currently least to greatest
        return num

    # def reset_parameters(self):
    #     relu_gain = nn.init.calculate_gain('relu')
    #     nn.init.uniform_(self.action_probs.weight.data, 0, .1 / self.num_outputs * 2)
    #     nn.init.uniform_(self.QFunction.weight.data, 0, .1 / self.basis_size)

    def forward(self, inputs):
        x = self.basis_fn(inputs) # TODO: dimension
        # print(self.basis_matrix.shape, x.shape)
        # print("xprebasis", x, self.basis_matrix)
        # print(x.shape, self.basis_matrix.shape)
        x = torch.mm(x, self.basis_matrix)
        # print("xbasis", x.shape)
        Qvals = self.QFunction(x)
        aprobs = self.action_probs(x)
        values = Qvals.max(dim=1)[0]
        probs = F.softmax(aprobs, dim=1)
        # print("probs", probs)
        log_probs = F.log_softmax(aprobs, dim=1)

        dist_entropy = -(log_probs * probs).sum(-1).mean()

        # print("xval", x)
        return values, dist_entropy, aprobs, Qvals

class FourierBasisModel(BasisModel):
    def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=None, sess = None):
        super().__init__(args, num_inputs, num_outputs, name=name, factor=factor, minmax=minmax, sess = None)
        '''
        factor is the order
        layers defines the variate (1 = univariate, 2 = paired, 3=all)
        object_extractors[0] is the current object
        the remainder are any correlate objects, with relative computations 
        computations are relative to pre_extracted state (just getters)
        '''
        self.order_vector = []
        for i in range (self.order):
            self.order_vector.append(np.pi*2*i/self.period)
        self.order_vector = pytorch_model.wrap(np.array(self.order_vector))
        self.order_vector.requires_grad = False
        self.train()
        self.reset_parameters()
        # print("done initializing")


    def basis_fn(self, inputs):
        '''
        computes the basis function values, that is, goes from [batch, num inputs] -> [batch, num_inputs * order]
        '''
        # print("minmax", self.min_, self.max_)
        if self.minmax is not None and self.use_normalize:
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

class GaussianBasisModel(BasisModel):
    def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=None, sess=None):
        super(GaussianBasisModel, self).__init__(args, num_inputs, num_outputs, name=name, factor=factor, minmax=minmax, sess=None)
        '''
        factor is the order
        layers defines the variate (1 = univariate, 2 = paired, 3=all)
        object_extractors[0] is the current object
        the remainder are any correlate objects, with relative computations 
        computations are relative to pre_extracted state (just getters)
        period is reused to be the variance, but might be dynamically determined
        '''
        minvs, maxvs = self.minmax
        self.order_vectors = []
        for minv, maxv in zip(minvs, maxvs):
            order_vector = []
            for i in range (self.order):
                if not self.normalize:
                    order_vector.append((minv + i * (maxv - minv) / (self.order - 1)))
                else:
                    order_vector.append((i / (self.order - 1)))
            self.order_vectors.append(pytorch_model.wrap(np.array(order_vector)))
        for vec in self.order_vectors:
            vec.requires_grad = False   
        self.train()
        self.reset_parameters()
        # print("done initializing")

    def basis_fn(self, inputs):
        # print("minmax", self.min_, self.max_)
        if self.minmax is not None and self.use_normalize:
            inputs = self.normalize(inputs)
        # for loops are supposed to be bad practice, but we will just keep these for now
        bat = []
        # print("ord", self.order_vector)
        for datapt in inputs:
            basis = []
            for order_vector, val in zip(self.order_vectors, datapt):
                # print ("input single", val)
                # print(val - order_vector)
                basis.append(torch.exp(-(val - order_vector).pow(2)/(2*(self.period ** 2))))
                # print(basis)
                # print("ov, val", order_vector, val, torch.exp(-(val - order_vector).pow(2)/(2*self.period)))
            # print(basis)
            basis = torch.cat(basis)
            bat.append(basis)
        return torch.stack(bat)

class GaussianMultilayerModel(GaussianBasisModel):
    def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=None,sess = None):
        super().__init__(args, num_inputs, num_outputs, name=name, factor=factor, minmax=minmax,sess = None)
        '''
        num_population is used as the size of the last layer (don't mix evolutionary and gaussian basis for now)        
        '''
        self.l1 = nn.Linear(self.basis_size, args.num_population)
        self.QFunction = nn.Linear(args.num_population, self.num_outputs, bias=False)
        self.action_probs = nn.Linear(args.num_population, self.num_outputs, bias=False)
        self.layers[2] = self.QFunction
        self.layers[3] = self.action_probs
        self.layers.append(self.l1)
        # print("done initializing")

    def forward(self, inputs):
        x = self.basis_fn(inputs) # TODO: dimension
        # print(self.basis_matrix.shape, x.shape)
        # print("xprebasis", x, self.basis_matrix)
        # print(x.shape, self.basis_matrix.shape)
        x = torch.mm(x, self.basis_matrix)
        # print("xbasis", x.shape)
        x = self.l1(x)
        x = F.relu(x)
        Qvals = self.QFunction(x)
        aprobs = self.action_probs(x)
        values = Qvals.max(dim=1)[0]
        probs = F.softmax(aprobs, dim=1)
        # print("probs", probs)
        log_probs = F.log_softmax(aprobs, dim=1)

        dist_entropy = -(log_probs * probs).sum(-1).mean()

        # print("xval", x)
        return values, dist_entropy, aprobs, Qvals
