import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from Models.models import Model, pytorch_model, models
from file_management import default_value_arg

# class SoftmaxModel(nn.Module):
#     def __init__(self, num_inputs, num_outputs, hidden_size):
#         super().__init__(self)
#         self.l1 = nn.Linear(num_inputs, hidden_size)
#         self.l2 = nn.Linear(hidden_size, num_outputs)

#     def forward(self, inputs):
#         x = self.l1(inputs)
#         out = self.l2(x)

class ObjectVectorModel(Model): 
    '''
    Simplest variable input network
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        args, num_inputs, num_outputs, factor = self.get_args(kwargs)
        self.class_sizes = default_value_arg(kwargs, 'class_sizes', [])
        self.key_dim = args.key_dim

        print(self.class_sizes, self.key_dim)
        self.key_map = [nn.Linear(self.class_sizes[i], self.key_dim) for i in range(len(self.class_sizes))]
        self.key_map = nn.ModuleList(self.key_map)
        self.layers.append(self.key_map)
        self.reduce_function = torch.mean # TODO: other transform functions (max pooling?)
        kwargs['num_inputs'] = self.key_dim
        if args.post_transform_form == 'none':
            self.post_network = None
            self.insize = self.key_dim
            self.init_last(num_outputs)
        else:
            kwargs['needs_final'] = False
            self.post_network = models[args.post_transform_form](**kwargs)
            self.layers.append(self.post_network)
        self.train()
        self.reset_parameters()

    def hidden(self, x, resp):
        maps = []
        # for s, res in zip(x, resp): # for each in batch
        #     lr = 0
        #     curmap = []
        #     for i, r in enumerate(res): # for each resp
        #         r = r.long().squeeze()
        #         # print(lr, r, resp)
        #         cx = s[lr:r] # state for one class
        #         # print(self.class_sizes)
        #         l = self.class_sizes[i]
        #         # print(l, r, lr, cx, )
        #         # print(list(range((r-lr) // l)))
        #         cv = [self.acti(self.key_map[i](ov)) for ov in [cx[v*l:(v+1)*l] for v in range((r-lr) // l)]] # l should evenly divide
        #         # print(r, cv)
        #         curmap += cv
        #         lr = r
        #     curmap = torch.stack(curmap)
        #     curmap = self.reduce_function(curmap, dim = 0)
        #     maps.append(curmap)
        # x = torch.stack(maps, dim=0)

        # fixed resp code:
        lr = 0
        maps = []
        for i,r in enumerate(resp[0]):
            # print(resp[0],r, self.class_sizes)
            r = r.long().squeeze()
            l = self.class_sizes[i]
            cx = x[:, lr:r]
            maps.append(self.acti(self.key_map[i](cx)))
        # print(maps[0].shape, torch.stack(maps, dim = 1).shape, self.reduce_function(torch.stack(maps, dim = 1), dim=1).shape)
        x = self.reduce_function(torch.stack(maps, dim = 1), dim=1)

        # x = self.acti(x)
        # print(x.shape)
        if self.post_network is not None:
            x = self.post_network.hidden(x, resp)
        return x

    # def forward(self, x, resp):
    #     x = self.hidden(x, resp)
    #     values, dist_entropy, probs, Q_vals = super().forward(x)
    #     return values, dist_entropy, probs, Q_vals
def torchmaxval(x, dim):
    return torch.max(x, dim=dim)[0]

class VariableInputAttentionModel(Model): 
    '''
    Model that takes 
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        args, num_inputs, num_outputs, factor = self.get_args(kwargs)
        self.class_sizes = default_value_arg(kwargs, 'class_sizes', [])
        self.key_dim = args.key_dim
        self.value_dim = args.value_dim

        self.key_map = nn.ModuleList([nn.Linear(self.class_sizes[i], self.key_dim) for i in range(len(self.class_sizes))])
        self.query_map = nn.ModuleList([nn.Linear(self.class_sizes[i], self.key_dim) for i in range(len(self.class_sizes))])
        self.value_map = nn.ModuleList(sum([[nn.Linear(self.class_sizes[i], self.value_dim) for j in range(len(self.class_sizes))] for i in range(len(self.class_sizes))], []))
        # reduce reduces along the softmax, while combine reduces across different key objects 
        # TODO: differentiate between head and tail
        self.reduce_function = torch.sum # TODO: other transform functions (max pooling?)
        self.object_reduction = torchmaxval
        self.combine_function = torch.mean
        self.hidden_size = self.value_dim * (len(self.class_sizes) ** 2)
        self.post_network = None
        if args.post_transform_form == 'none':
            self.post_network = None
            self.insize = self.value_dim
            self.init_last(num_outputs)
        else:
            kwargs['num_inputs'] = self.hidden_size
            kwargs['needs_final'] = False
            self.post_network = models[args.post_transform_form](**kwargs)
        self.train()
        self.reset_parameters()

    def hidden(self, x, resp):
        batch_values = []
        for xv, res in zip(x, resp):
            lr = 0
            keys = []
            queries = []
            obj_vals = []
            for i, r in enumerate(res):
                r = r.long().squeeze()
                cx = xv[lr:r] # state for one class
                l = self.class_sizes[i]
                inp_vals = [cx[v*l:(v+1)*l] for v in range((r-lr) // l)] # l should evenly divide
                obj_vals.append(inp_vals)
                ck = [self.acti(self.key_map[i](ov)) for ov in inp_vals]
                cq = torch.stack([self.acti(self.query_map[i](ov)) for ov in inp_vals], dim = 0) # batch x num instances x key dim 
                keys.append(ck)                                                                                 
                queries.append(cq)
            # adapted transform. K,V from object instance A, q from object class B. combine(k*q) for all within all q of the same instance, and take the softmax of the values 
            queried_vals = []
            for i, (obv,obk) in enumerate(zip(obj_vals, keys)): # iterate over every input class
                for v,k in zip(obv, obk): # iterate over every instance in the input class
                    comp = []
                    values = []
                    for j, q in enumerate(queries):
                        # print(k.shape, q.shape)
                        comp.append(self.object_reduction(torch.mm(k.unsqueeze(0), q.t()), dim = 1))
                        values.append(self.acti(self.value_map[i * len(self.class_sizes) + j](v)))
                    softmax_vals = F.softmax(torch.stack(comp, dim=0), dim = 0) # num instances a x num instances b x 1
                    # print(self.reduce_function(torch.stack(values, dim=0) * softmax_vals, dim=0).shape, torch.stack(values, dim=0).shape)
                    queried_vals.append(self.reduce_function(torch.stack(values, dim=0) * softmax_vals, dim=0))
            vals = self.combine_function(torch.stack(queried_vals, dim=0), dim=0)
            batch_values.append(vals)
        x = torch.stack(batch_values, dim = 0)
        # print(x.shape)
        if self.post_network is not None:
            x = self.post_network.hidden(x)
        return x

    # def forward(self, x, resp):
    #     x = self.hidden(x, resp)
    #     values, dist_entropy, probs, Q_vals = super().forward(x)
    #     return values, dist_entropy, probs, Q_vals

class FixedInputAttentionModel(Model):    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        args, num_inputs, num_outputs, factor = self.get_args(kwargs)
        factor = int(factor)
        self.key_dim = args.key_dim
        self.value_dim = args.value_dim
        self.ikd = 1.0/np.sqrt(args.key_dim)
        self.hidden_size = args.value_dim * self.num_inputs
        print("Network Sizes: ", self.key_dim, self.insize, self.hidden_size)
        # self.l1 = nn.Linear(self.num_inputs, self.num_inputs*factor*factor)
        self.key_map = nn.Parameter(torch.zeros(self.num_inputs, self.key_dim))
        self.query_map = nn.Parameter(torch.zeros(self.num_inputs, self.key_dim))
        self.value_map = nn.Parameter(torch.zeros(self.num_inputs, self.value_dim))
        self.dk = self.key_dim
        kwargs['num_inputs'] = self.hidden_size
        self.post_network = None
        if args.post_transform_form == 'none':
            self.post_network = None
            self.insize = self.value_dim * self.num_inputs
            self.init_last(num_outputs)
        else:
            kwargs['num_inputs'] = self.hidden_size
            kwargs['needs_final'] = False
            self.post_network = models[args.post_transform_form](**kwargs)
            self.layers.append(post_network)
        self.layers += [self.key_map, self.query_map, self.value_map]

        self.train()
        self.reset_parameters()

    def attention(self, x):
        # if self.minmax is not None and self.use_normalize:
        #     x = self.normalize(x)
        # x = x * self.scale
        x = torch.stack([torch.diag(xs) for xs in x]) # batch x input size x input size
        # print(x.shape)
        # print(self.key_map)
        k = torch.matmul(x, self.key_map) # batch, input_size, input size x input size, key_dim
        # print(k.shape)
        q = torch.matmul(x, self.query_map) # batch, input_size, input size x input size, key_dim
        # print(q.shape)
        a = torch.bmm(k, q.transpose(1,2)) / self.dk # batch, input_size, key dim x batch, key_dim, input size
        # print(a)
        x = torch.matmul(F.softmax(a, dim=2).transpose(1,2), self.value_map) # batch, input_size, input size -> batch, 1, input_size x input_size x value dim
        # x = x.mean(dim=1) # reduce along the keys so a single value is output
        # print(F.softmax(a, dim=1).transpose_(1,2))
        # print(x.shape)
        x = x.view(-1, self.hidden_size)
        return x


    def hidden(self, x, resp):
        x = self.attention(x)
        if self.post_network is not None:
            x = self.post_network.hidden(x)
        return x

    # def forward(self, x, resp):
    #     x = self.hidden(x, resp)
    #     values, dist_entropy, probs, Q_vals = super().forward(x)
    #     return values, dist_entropy, probs, Q_vals

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

class ObjectAttentionModel(Model): 
    '''
    Simplest variable input network
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        args, num_inputs, num_outputs, factor = self.get_args(kwargs)
        self.class_sizes = default_value_arg(kwargs, 'class_sizes', [])
        self.key_dim = args.key_dim
        self.value_dim = args.value_dim
        self.dk = self.key_dim
        print(self.class_sizes, self.key_dim)
        self.key_map = [nn.Linear(self.class_sizes[i], self.key_dim) for i in range(len(self.class_sizes))]
        self.key_map = nn.ModuleList(self.key_map)
        self.layers.append(self.key_map)
        self.query_map = [nn.Linear(self.class_sizes[i], self.key_dim) for i in range(len(self.class_sizes))]
        self.query_map = nn.ModuleList(self.query_map)
        self.layers.append(self.query_map)
        self.value_map = [nn.Linear(self.class_sizes[i], self.value_dim) for i in range(len(self.class_sizes))]
        self.value_map = nn.ModuleList(self.value_map)
        self.layers.append(self.value_map)
        self.reduce_function = torch.mean # TODO: other transform functions (max pooling?)
        self.hidden_dim = len(self.class_sizes) * self.value_dim
        kwargs['num_inputs'] = self.hidden_dim
        if args.post_transform_form == 'none':
            self.post_network = None
            self.insize = self.hidden_dim
            self.init_last(num_outputs)
        else:
            kwargs['needs_final'] = False
            self.post_network = models[args.post_transform_form](**kwargs)
            self.layers.append(self.post_network)
        self.train()
        self.reset_parameters()

    def hidden(self, x, resp):
        keys = []
        queries = []
        values = []
        # fixed resp code:
        lr = 0
        for i,r in enumerate(resp[0]):
            # print(resp[0],r, self.class_sizes)
            r = r.long().squeeze()
            l = self.class_sizes[i]
            cx = x[:, lr:r]
            keys.append(self.acti(self.key_map[i](cx)))
            queries.append(self.acti(self.query_map[i](cx)))
            values.append(self.acti(self.value_map[i](cx)))
        # print(maps[0].shape, torch.stack(maps, dim = 1).shape, self.reduce_function(torch.stack(maps, dim = 1), dim=1).shape)
        # x = self.reduce_function(torch.stack(maps, dim = 1), dim=1)
        a = torch.bmm(torch.stack(keys,dim=1), torch.stack(queries,dim=1).transpose(1,2)) / self.key_dim # batch, input_size, key dim x batch, key_dim, input size
        # print(a)
        x = torch.matmul(F.softmax(a, dim=2).transpose(1,2), torch.stack(values, dim=1)) # batch, input_size, input size -> batch, 1, input_size x input_size x value dim
        # x = x.mean(dim=1) # reduce along the keys so a single value is output
        # print(F.softmax(a, dim=1).transpose_(1,2))
        # print(x.shape)
        x = x.view(-1, self.hidden_dim)

        # x = self.acti(x)
        # print(x.shape)
        if self.post_network is not None:
            x = self.post_network.hidden(x, resp)
        return x

class MultiHeadedModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        args, num_inputs, num_outputs, factor = self.get_args(kwargs)
        factor = int(args.factor)
        kwargs['needs_final'] = False
        self.num_heads = args.num_heads
        self.attentions = nn.ModuleList([models[args.attention_form](**kwargs) for _ in range(args.num_heads)])
        kwargs['num_inputs'] = self.attentions[0].insize * args.num_heads
        kwargs['factor'] = args.factor
        kwargs['num_layers'] = args.num_layers
        # if args.post_transform_form == 'none':
        #     self.post_network = None
        self.insize = self.attentions[0].insize * args.num_heads
        self.init_last(num_outputs)
        # else:
        #     kwargs['num_inputs'] = self.hidden_size
        #     kwargs['needs_final'] = False
        #     self.post_network = models[args.post_transform_form](**kwargs)
        self.train()
        self.reset_parameters()

    def hidden(self, x, resp):
        x = torch.cat([attention.hidden(x, resp) for attention in self.attentions], dim=1)
        # if self.post_network is not None:
        #     x = self.post_network.hidden(x, resp) # TODO: the meaning of resp is invalid here, so post network should not use resp...
        return x

    # def forward(self, x, resp):
    #     x = self.hidden(x, resp)
    #     values, dist_entropy, probs, Q_vals = super().forward(x)
    #     return values, dist_entropy, probs, Q_vals


attention_networks = {'fixedattention': FixedInputAttentionModel, 'vector': ObjectVectorModel, 'attention': VariableInputAttentionModel, 'objectattention': ObjectAttentionModel}
models = {**models, **attention_networks}