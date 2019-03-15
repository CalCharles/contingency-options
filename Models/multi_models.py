from Models.models import Model, models
import torch.nn as nn

class PopulationModel(Model):
	def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=None, sess = None):
		super(MLPEvolutionPolicy, self).__init__(self, args, num_inputs, num_outputs, name=name, factor=factor, minmax=minmax, sess=sess)
		self.layers = []
		networks = []
		print("initializing options")
		for i in range(args.num_population):
			networks.append(models[args.evolve_form](self, args, num_inputs, num_outputs, name=name, factor=factor, minmax=minmax, sess=sess))
		self.networks = nn.ModuleList(networks)
		self.num_population = args.num_population
		self.name = name + "evol"

		# if running in serial
		self.run_duration = args.sample_duration
		self.current_network_index = 0

	def set_networks(self, networks):
		self.networks = nn.ModuleList(networks)
		# print(self.state_dict())
		self.current_network_index = 0
		self.current_run_duration = self.run_duration

	def reset_parameters(self):
		for network in self.networks:
			network.reset_parameters()

	def forward(self, inputs):
		return self.networks[self.current_network_index](inputs)
