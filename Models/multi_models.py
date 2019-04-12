from Models.models import Model, models
import torch.nn as nn

class PopulationModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        args, num_inputs, num_outputs, factor = self.get_args(kwargs)
        self.layers = []
        networks = []
        print("initializing options")
        for i in range(args.num_population):
            networks.append(models[args.base_form](**kwargs))
        self.networks = nn.ModuleList(networks)
        self.mean = models[args.base_form](**kwargs)
        self.best = models[args.base_form](**kwargs)
        # self.networks = networks
        self.layers += self.networks
        self.num_population = args.num_population

        # if running in serial
        self.run_duration = args.sample_duration
        self.current_network_index = 0
        self.reset_parameters()
        self.use_mean = False

    def currentModel(self):
        return self.networks[self.current_network_index]

    def cuda(self):
        for network in self.networks:
            network.cuda()
        return super().cuda()


    def cpu(self):
        for network in self.networks:
            network.cpu()
        return super().cpu()


    def set_networks(self, networks):
        self.networks = nn.ModuleList(networks)
        # self.networks = networks
        # print(self.state_dict())
        self.current_network_index = 0
        self.current_run_duration = self.run_duration

    def reset_parameters(self):
        for network in self.networks:
            network.reset_parameters()

    def hidden(self, inputs, resp, idx=-1):
        if self.test or self.use_mean:
            return self.mean.hidden(inputs, resp)
        if idx < 0:
            return self.networks[self.current_network_index].hidden(inputs, resp)
        return self.networks[idx].hidden(inputs, resp)        

    def forward(self, inputs, resp, idx=-1):
        # self.current_network_index = (self.current_network_index + 1) % self.num_population
        if self.test or self.use_mean:
            return self.mean(inputs, resp)
        if idx < 0:
            return self.networks[self.current_network_index](inputs, resp)
        return self.networks[idx](inputs, resp)