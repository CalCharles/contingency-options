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
    def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=None):
        super(Model, self).__init__()
        num_inputs = int(num_inputs)
        num_outputs = int(num_outputs)
        self.insize = max(num_inputs * num_outputs * factor * factor, num_inputs * num_outputs)
        self.minmax = minmax
        if minmax is not None:
            self.minmax = (torch.cat([pytorch_model.wrap(minmax[0]).cuda() for _ in range(args.num_stack)], dim=0), torch.cat([pytorch_model.wrap(minmax[1]).cuda() for _ in range(args.num_stack)], dim=0))
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.layers = []
        self.critic_linear = nn.Linear(self.insize, 1)
        self.time_estimator = nn.Linear(self.insize, 1)
        self.QFunction = nn.Linear(self.insize, num_outputs)
        self.action_probs = nn.Linear(self.insize, num_outputs)
        self.name = name
        self.iscuda = args.cuda # TODO: don't just set this to true
        Model.reset_parameters(self)

    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('relu')
        # nn.init.uniform_(self.critic_linear.weight.data, .9 / self.insize, 1.1 / self.insize)
        # nn.init.uniform_(self.time_estimator.weight.data, .9 / self.insize, 1.1 / self.insize)
        # nn.init.uniform_(self.QFunction.weight.data, .9 / self.insize, 1.1 / self.insize)
        # nn.init.uniform_(self.action_probs.weight.data, .9 / self.insize, 1.1 / self.insize)
        nn.init.uniform_(self.critic_linear.weight.data, 0.0, 1.1 / self.insize)
        nn.init.uniform_(self.time_estimator.weight.data, 0.0, 1.1 / self.insize)
        nn.init.uniform_(self.QFunction.weight.data, 0.0, 1.1 / self.insize)
        nn.init.uniform_(self.action_probs.weight.data, 0.0, 1.1 / self.insize)
        nn.init.uniform_(self.critic_linear.bias.data, 0.0,.1)
        nn.init.uniform_(self.time_estimator.bias.data, 0.0,.1)
        nn.init.uniform_(self.QFunction.bias.data, 0.0,.1)
        nn.init.uniform_(self.action_probs.bias.data, 0.0,.1)

        # nn.init.uniform_(self.critic_linear.bias.data, -.1,.1)
        # nn.init.uniform_(self.time_estimator.bias.data, -.1,.1)
        # nn.init.uniform_(self.QFunction.bias.data, -.1,.1)
        # nn.init.uniform_(self.action_probs.bias.data, -.1,.1)

    def normalize(self, x):
        return (x - self.minmax[0]) / (self.minmax[1] - self.minmax[0])

    def forward(self, x):
        '''
        TODO: make use of time_estimator?, link up Q vals and action probs
        TODO: clean up cuda = True to something that is actually true
        input: [batch size, state size] (TODO: no multiple processes)
        output [batch size, 1], [batch size, 1], [batch_size, num_actions], [batch_size, num_actions]
        '''
        if self.minmax is None:
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

    def save(self, pth):
        torch.save(self, pth + self.name + ".pt")

class BasicModel(Model):
    def __init__(self, args, num_inputs, num_outputs, name="option", minmax=None):
        super(BasicModel, self).__init__(args, num_inputs, num_outputs, name=name, factor=args.factor, minmax=minmax)
        factor = int(args.factor)
        self.hidden_size = self.num_inputs*factor*factor
        print("Network Sizes: ", self.num_inputs, self.num_inputs*factor*factor, max(self.num_inputs*self.num_outputs * factor, self.num_inputs * self.num_outputs))
        # self.l1 = nn.Linear(self.num_inputs, self.num_inputs*factor*factor)
        self.l1 = nn.Linear(self.num_inputs,self.insize)

        self.l2 = nn.Linear(self.num_inputs*factor*factor, self.insize)
        self.layers.append(self.l1)
        self.layers.append(self.l2)
        super().reset_parameters()
        self.train()
        self.reset_parameters()


    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('relu')
        # nn.init.uniform_(self.l1.weight.data, .9 / self.num_inputs, 1.1 / self.num_inputs)
        # nn.init.uniform_(self.l2.weight.data, .9 / self.hidden_size, 1.1 / self.hidden_size)
        nn.init.uniform_(self.l1.weight.data, 0.0, 1.1 / self.num_inputs / 10.0)
        nn.init.uniform_(self.l2.weight.data, 0.0, 1.1 / self.hidden_size / 10.0)
        nn.init.uniform_(self.l1.bias.data, 0.0,.1)
        nn.init.uniform_(self.l2.bias.data, 0.0,.1)

    def forward(self, x):
        '''
        TODO: make use of time_estimator, link up Q vals and action probs
        TODO: clean up cuda = True to something that is actually true
        '''
        # print(x.shape)
        if self.minmax is None:
            pass
        else:
            x = self.normalize(x)
        x = self.l1(x)
        x = F.relu(x)
        # x = self.l2(x)
        # x = F.relu(x)
        # print(self.l2.weight)
        # print(x.shape)
        action_probs = self.action_probs(x)
        Q_vals = self.QFunction(x)
        # print(self.action_probs.weight)
        values = self.critic_linear(x)
        probs = F.softmax(action_probs, dim=1)
        log_probs = F.log_softmax(action_probs, dim=1)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        # print("lp, p", action_probs, log_probs, probs)
        # print(values.shape, probs.shape, dist_entropy.shape, Q_vals.shape)

        return values, dist_entropy, probs, Q_vals

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

class FourierOptionPolicy(Model):
    def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=None):
        super(FourierOptionPolicy, self).__init__(args, num_inputs, num_outputs, name=name, factor=factor, minmax=minmax)
        '''
        factor is the order
        layers defines the variate (1 = univariate, 2 = paired, 3=all)
        object_extractors[0] is the current object
        the remainder are any correlate objects, with relative computations 
        computations are relative to pre_extracted state (just getters)
        '''
        self.order = factor
        self.variate = args.num_layers
        self.layers= []
        self.period = args.eps
        self.base_vals = [i for i in range(self.order+1)]

        self.order_vector = []
        for i in range (self.order + 1):
            self.order_vector.append(np.pi*2*i/self.period)
        self.order_vector = pytorch_model.wrap(np.array(self.order_vector))
        print("VARIATE", self.variate)
        if self.variate == 1:
            self.basis_matrix = pytorch_model.wrap(np.identity(self.num_inputs * (self.order + 1)))
            self.basis_size = (self.order + 1) * self.num_inputs
            self.QFunction = nn.Linear((self.order + 1) * self.num_inputs, self.num_outputs, bias=False)
        elif num_layers == 2:
            # TODO: use HIST LEN = num stack to get groupings
            print( "not implemented yet")
        elif num_layers == 3:
            # print((self.order + 1), self.num_inputs)
            # print((self.order + 1) ** (self.num_inputs))
            self.basis_matrix = np.zeros(((self.num_inputs) * (self.order + 1), (self.order + 1) ** (self.num_inputs)))
            bases_indexes = []
            for i in range((self.order + 1) ** (self.num_inputs)):
                bases = self.num_to_base(i)
                bases = [b + (self.order + 1) * i for i, b in enumerate(bases)]
                bases_indexes.append(bases)
            # print(self.basis_matrix.shape)
            indexes = np.array([[i for _ in range(self.num_inputs)] for i in range((self.order + 1) ** (self.num_inputs))]).flatten()
            self.basis_matrix = self.basis_matrix.T
            self.basis_matrix[indexes, np.array(bases_indexes).flatten()] = 1
            self.basis_matrix = self.basis_matrix.T
            self.basis_matrix = pytorch_model.wrap(self.basis_matrix)
            self.basis_size = (self.order + 1) ** (self.num_inputs)
            self.QFunction = nn.Linear((self.order + 1) ** (self.num_inputs), self.num_outputs, bias=False)

        # print("order vector", self.order_vector)
        # self.time_estimator = nn.Linear(max(num_inputs*num_outputs * factor, num_inputs * num_outputs), 1)
        self.action_probs = nn.Linear(self.num_outputs, self.num_outputs) # TODO: action propbs are based on Q values

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


models = {"basic": BasicModel, "tab": TabularQ, "fourier": FourierOptionPolicy}
