

class ObjectVectorModel(Model): 
    '''
    Model that takes 
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        args, num_inputs, num_outputs, factor = self.get_args(kwargs)
        self.class_sizes = default_value_arg(kwargs, 'class_sizes', [])
        self.key_dim = args.key_dim

        self.key_map = [nn.Linear(class_sizes[i], self.key_dim) for i in range(len(self.class_sizes))]
        self.reduce_function = torch.mean # TODO: other transform functions (max pooling?)
        kwargs['num_inputs'] = self.key_dim
        self.post_network = self.models[args.post_transform_form](**kwargs)
        self.train()
        self.reset_parameters()

    def hidden(self, x, resp):
        lr = 0
        maps = []
        for i, r in enumerate(resp):
            cx = x[:, lr:r] # state for one class
            l = self.class_sizes[i]
            cv = [self.key_map[i](ov) for ov in [cx[:, v*l:(v+1)*l] for v in range((r-lr) // l)]] # l should evenly divide
            maps += cv
        maps = torch.stack(maps, dim = 1)
        x = self.reduce_function(maps, dim = 1)
        x = self.post_network.hidden(x)
        return x

    # def forward(self, x, resp):
    #     x = self.hidden(x, resp)
    #     values, dist_entropy, probs, Q_vals = super().forward(x)
    #     return values, dist_entropy, probs, Q_vals

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

        self.key_map = nn.ModuleList([nn.Linear(class_sizes[i], self.key_dim) for i in range(len(self.class_sizes))])
        self.query_map = nn.ModuleList([nn.Linear(class_sizes[i], self.key_dim) for i in range(len(self.class_sizes))])
        self.value_map = nn.ModuleList([[nn.Linear(class_sizes[i] + class_sizes[j], self.value_dim) for j in range(len(self.class_sizes))] for i in range(len(self.class_sizes))])
        # reduce reduces along the softmax, while combine reduces across different key objects 
        # TODO: differentiate between head and tail
        self.reduce_function = torch.sum # TODO: other transform functions (max pooling?)
        self.combine_function = torch.mean
        self.hidden_size = self.value_dim * (len(self.class_sizes) ** 2)
        kwargs['num_inputs'] = self.hidden_size
        self.post_network = self.models[args.post_transform_form](**kwargs)
        self.train()
        self.reset_parameters()

    def hidden(self, x, resp):
        batch_values = []
        for xv in x
            lr = 0
            keys = []
            queries = []
            obj_vals = []
            for i, r in enumerate(resp):
                cx = xv[lr:r] # state for one class
                inp_vals = [cx[v*l:(v+1)*l] for v in range((r-lr) // l)] # l should evenly divide
                obj_vals.append(inp_vals)
                ck = torch.stack([self.key_map[i](ov) for ov in inp_vals], dim = 0)
                cq = torch.stack([self.query_map[i](ov) for ov in inp_vals], dim = 0) # batch x num instances x key dim 
                keys.append(ck)                                                                                 
                queries.append(cq)
            vals = []
            for i, (av, ak, aq) in enumerate(zip(obj_vals, keys, queries)):
                for j, (bv, bk, bq) in enumerate(zip(obj_vals, keys, queries)):
                    raw_values = torch.stack([torch.stack([self.value_map[i][j](torch.cat((v1, v2), dim=0)) for v2 in bv], dim = 0) for v1 in av]) # num instances a x num instances b x value size
                    softmax_vals = F.softmax(torch.mm(ak, bq.t()), dim = 0) # num instances a x num instances b x 1
                    vals.append(self.combine_function(self.reduce_function(raw_values * softmax_vals, dim=0), dim=0)) # reduce over both keys: value_dim
            batch_values.append(torch.cat(vals))
        x = torch.stack(batch_values, dim = 1)
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
        self.key_map = []
        self.query_maps = []
        self.value_maps = []
        self.key_map = nn.Parameter(torch.zeros(self.num_inputs, self.key_dim))
        self.query_map = nn.Parameter(torch.zeros(self.num_inputs, self.key_dim))
        self.value_maps = nn.Parameter(torch.zeros(self.num_inputs, self.value_dim))
        self.dk = self.key_dim
        kwargs['num_inputs'] = self.hidden_size
        self.post_network = models[args.post_transform_form](**kwargs)

        self.train()
        self.reset_parameters()

    def attention(self, x):
        # if self.minmax is not None and self.use_normalize:
        #     x = self.normalize(x)
        # x = x * self.scale
        x = torch.stack([torch.diag(xs) for xs in x]) # batch x input size x input size
        k = torch.mm(x, self.key_map) # batch x input_size, key_dim
        q = torch.mm(x, self.query_map) # batch x input_size, key_dim
        a = torch.mm(k, q.t()) / self.dk # batch x input_size, input_size
        x = torch.mm(F.softmax(a), self.value_maps) # batch x input_size, hidden_size
        x = x.view(-1, self.hidden_size)
        return x


    def hidden(self, x, resp):
        x = self.attention(x)
        x = self.post_network(x)
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

class MultiHeadedAttention(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        args, num_inputs, num_outputs, factor = self.get_args(kwargs)
        factor = int(args.factor)
        self.attentions = nn.ModuleList([attention_networks[args.attention_form](**kwargs) for _ in range(args.num_heads)])
        kwargs['num_inputs'] = self.attentions[0].insize * num_heads
        kwargs['factor'] = args.post_factor
        kwargs['num_layers'] = args.post_layers
        self.post_network = models[args.post_transform_form](**kwargs)
        self.train()
        self.reset_parameters()

    def hidden(self, x, resp):
        x = torch.cat([attention.hidden(x) for attention in self.attentions], dim=1)
        x = self.post_network.hidden(x)
        return x

    # def forward(self, x, resp):
    #     x = self.hidden(x, resp)
    #     values, dist_entropy, probs, Q_vals = super().forward(x)
    #     return values, dist_entropy, probs, Q_vals


attention_networks = {'fixed': FixedInputAttentionModel, 'vector': ObjectVectorModel, 'vary': VariableInputAttentionModel}