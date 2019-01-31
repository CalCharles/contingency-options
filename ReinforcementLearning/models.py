import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


class pytorch_model():
    def __init__(self, combiner=None, loss=None, reducer=None, cuda=True):
        # should have customizable combiner and loss, but I dont.
        self.cuda=cuda
        self.reduce_size = 2 # someday this won't be hard coded either

    @staticmethod
    def wrap(data, dtype=torch.float, cuda=True):
        # print(Variable(torch.Tensor(data).cuda()))
        if cuda:
            return Variable(torch.tensor(data, dtype=dtype).cuda())
        else:
            return Variable(torch.tensor(data, dtype=dtype))

    @staticmethod
    def unwrap(data):
        return data.data.cpu().numpy()

    @staticmethod
    def concat(data, axis=0):
        return torch.cat(data, dim=axis)

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
        self.iscuda = args.cuda # TODO: don't just set this to true

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

class BasicModel(Model):
    def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=(-1,-1)):
        super(BasicModel, self).__init__(args, num_inputs, num_outputs, name="option", factor=8, minmax=(-1,-1))
        factor = int(factor)
        self.l1 = nn.Linear(self.num_inputs, self.num_inputs*factor)
        self.l2 = nn.Linear(self.num_inputs*factor, max(self.num_inputs*self.num_outputs * factor, self.num_inputs * self.num_outputs))
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
        # print(x.shape)
        action_probs = self.action_probs(x)
        Q_vals = self.QFunction(x)
        values = self.critic_linear(x)
        probs = F.softmax(action_probs, dim=1)
        log_probs = F.log_softmax(action_probs, dim=1)

        dist_entropy = -(log_probs * probs).sum(-1).mean()

        return values, dist_entropy, probs, Q_vals

class TabularQ(Model):
    def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=(-1,-1)):
        super(TabularQ, self).__init__(args, num_inputs, num_outputs, name="option", factor=8, minmax=(-1,-1))
        self.Qtable = dict()
        self.action_prob_table = dict()
        self.initial_value = 0.0 # need to set this
        self.initial_aprob = 1 / num_outputs

    def normalize(self, x):
        return (x - self.minmax[0]) / (self.minmax[1] - self.minmax[0])

    def forward(self, x):
        '''
        TODO: make use of time_estimator, link up Q vals and action probs
        TODO: clean up cuda = True to something that is actually true
        TODO: only accepts integer array input states of form (num in batch, num_vals)
        '''
        Qvals = []
        aprobs = []
        for xv in x: # for each x in the batch, convert state to hash and get Q value
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
        values = Q_vals.max(dim=1)
        probs = F.softmax(action_probs, dim=1)
        # print("probs", probs)
        log_probs = F.log_softmax(action_probs, dim=1)

        dist_entropy = -(log_probs * probs).sum(-1).mean()

        return values, dist_entropy, probs, Q_vals

class FourierOptionPolicy(COptPolicy):
    def __init__(self, num_inputs, hist_len, state_sizes, num_outputs, desired_mode, num_layers=1, factor=2, period=2, num_population = 10, run_duration=150, serial=False, sample_duration=100, object_extractors=[], sub_options=[]):
        '''
        factor is the order
        layers defines the variate (1 = univariate, 2 = paired, 3=all)
        object_extractors[0] is the current object
        the remainder are any correlate objects, with relative computations 
        computations are relative to pre_extracted state (just getters)
        '''
        super(FourierOptionPolicy, self).__init__(num_inputs, hist_len, state_sizes, num_outputs, desired_mode, object_extractors = object_extractors, sub_options=sub_options, serial=serial)

        self.order = factor
        self.variate = num_layers
        self.layers= []
        self.period = period
        self.base_vals = [i for i in range(self.order+1)]

        self.order_vector = []
        for i in range (self.order + 1):
            self.order_vector.append(np.pi*2*i/self.period)
        self.order_vector = pytorch_model.wrap(np.array(self.order_vector))
        if self.variate == 1:
            self.basis_matrix = []
            for i in range(self.num_inputs):
                basis_row = []
                for j in range(self.num_inputs):
                    if i == j:
                        basis_row.append(np.identity(self.order + 1))
                    else:
                        basis_row.append(np.zeros((self.order + 1, self.order + 1)))
                basis_row = np.hstack(basis_row)
                self.basis_matrix.append(basis_row)
            self.basis_matrix = pytorch_model.wrap(np.vstack(self.basis_matrix))
            self.weights = nn.Linear((self.order + 1) * self.num_inputs, self.num_outputs, bias=False)

        elif num_layers == 2:
            # TODO: use HIST LEN = num stack to get groupings
            print( "not implemented yet")
        elif num_layers == 3:
            print((self.order + 1), self.num_inputs)
            print((self.order + 1) ** (self.num_inputs))
            self.basis_matrix = np.zeros(((self.num_inputs) * (self.order + 1), (self.order + 1) ** (self.num_inputs)))
            bases_indexes = []
            for i in range((self.order + 1) ** (self.num_inputs)):
                bases = self.num_to_base(i)
                # print(bases)
                bases = [b + (self.order + 1) * i for i, b in enumerate(bases)]
                bases_indexes.append(bases)
            # print(bases_indexes)
            print(self.basis_matrix.shape)
            indexes = np.array([[i for _ in range(self.num_inputs)] for i in range((self.order + 1) ** (self.num_inputs))]).flatten()
            # print(indexes)
            self.basis_matrix = self.basis_matrix.T
            self.basis_matrix[indexes, np.array(bases_indexes).flatten()] = 1
            self.basis_matrix = self.basis_matrix.T
            self.basis_matrix = pytorch_model.wrap(self.basis_matrix)
            self.weights = nn.Linear((self.order + 1) ** (self.num_inputs), self.num_outputs, bias=False)

        print("order vector", self.order_vector)
        self.critic_linear = torch.mean
        self.time_estimator = nn.Linear(max(num_inputs*num_outputs * factor // 8, num_inputs * num_outputs), 1)
        # if action_space.__class__.__name__ == "Discrete":
        # num_outputs = action_space.n
        self.dist = Categorical(num_outputs, num_outputs, noforward=True)
        # elif action_space.__class__.__name__ == "Box":
        #   num_outputs = action_space.shape[0]
        # self.dist = DiagGaussian(num_inputs * num_outputs * 4, num_outputs)
        # elif action_space.__class__.__name__ == "int":
        # self.dist = Categorical(num_inputs * num_outputs * 4, action_space)
        # else:
        #   raise NotImplementedError
        # print("setting parameters")
        self.train()
        self.reset_parameters()
        # print("done initializing")

    def num_to_base(self, val):
        num = []
        base = self.order + 1
        for _ in range(self.num_inputs):
            # print(val % base, val, base)
            num.append(val % base)
            val = val // base
        # num.reverse() # currently least to greatest
        return num

    def reset_parameters(self):
        self.apply(rand_init)

        relu_gain = nn.init.calculate_gain('relu')
        self.weights.weight.data.mul_(.01)
        print("after reset", self.weights.weight.data)

    def fourier_basis(self, inputs):
        # print("minmax", self.min_, self.max_)
        if self.normalize:
            inputs = (inputs - self.min_) / self.max_ 
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
        x = self.weights(x.view(x.shape[0],-1))
        # print("xval", x)
        return x.max(dim=1)[0], x

    def time_estimate(self, inputs):
        cl, x = self.forward(inputs)
        return self.time_estimator(x)


models = {"basic": BasicModel, "tab": TabularQ}
