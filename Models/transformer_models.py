
# TODO: completely incomplete!!!!!!!!!!!!!!!!!!!!!!!!!!!
class AttentionModel(Model):    
    def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=None, sess=None, param_dim=-1):
        super().__init__(args, num_inputs, num_outputs, name=name, factor=factor, minmax=minmax, sess=None, param_dim=param_dim)
        factor = int(args.factor)
        self.key_dim = args.key_dim
        self.key_num = args.key_num
        self.ikd = 1.0/np.sqrt(args.key_dim)
        self.hidden_size = args.value_dim
        print("Network Sizes: ", self.key_dim, self.insize, self.hidden_size)
        # self.l1 = nn.Linear(self.num_inputs, self.num_inputs*factor*factor)
        self.key_map = []
        self.query_maps = []
        self.value_maps = []
        self.key_map = nn.Parameter(torch.zeros(self.num_inputs, self.key_dim))
        self.query_map = nn.Parameter(torch.zeros(self.num_inputs, self.key_dim))
        self.value_maps = nn.Parameter(torch.zeros(self.num_inputs, self.hidden_size))

        if args.num_layers == 1:
            self.l1 = nn.Linear(self.hidden_size,self.insize)
        elif args.num_layers == 2:
            self.l1 = nn.Linear(self.hidden_size,self.hidden_size)
            self.l2 = nn.Linear(self.hidden_size, self.insize)
        elif args.num_layers == 3:
            self.l1 = nn.Linear(self.hidden_size,self.hidden_size)
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


    def forward(self, x):
        '''
        TODO: make use of time_estimator, link up Q vals and action probs
        TODO: clean up cuda = True to something that is actually true
        '''
        # print(x.shape)
        if self.minmax is not None and self.use_normalize:
            x = self.normalize(x)
        x = x * self.scale
        x = torch.diag(x)
        k = torch.mm(self.key_map, x)
        q = torch.mm(self.query_map, x)
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
        if np.random.rand() < .001:
            print("total", x.sum().detach(), self.l1.weight.abs().sum().detach(), self.action_probs.weight.abs().sum().detach())
        values, dist_entropy, probs, Q_vals = super().forward(x)
        return values, dist_entropy, probs, Q_vals

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
