import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from ReinforcementLearning.models import pytorch_model
import numpy as np
from ReinforcementLearning.train_rl import sample_actions

class EpsilonGreedyQ():

    def initialize(self, args, num_outputs):
        self.epsilon = args.greedy_epsilon
        self.num_outputs = num_outputs

    def take_action(self, probs, q_vals):
        action = sample_actions(F.softmax(q_vals, dim=1), deterministic =True)
        if np.random.rand() < self.epsilon:
            action = pytorch_model.wrap(np.random.randint(self.num_outputs, size = q_vals.shape[0]), cuda = True)
        return action

    def set_test(self):
        self.epsilon = 0

class EpsilonStochasticQ():

    def initialize(self, args, num_outputs):
        self.epsilon = args.greedy_epsilon
        self.num_outputs = num_outputs

    def take_action(self, probs, q_vals):
        action = sample_actions(F.softmax(q_vals, dim=1), deterministic =False)
        if np.random.rand() < self.epsilon:
            action = pytorch_model.wrap(np.random.randint(self.num_outputs, size = q_vals.shape[0]), cuda = True)
        return action

    def set_test(self):
        self.epsilon = 0


class EpsilonGreedyProbs():

    def initialize(self, args, num_outputs):
        self.epsilon = args.greedy_epsilon
        self.num_outputs = num_outputs

    def take_action(self, probs, q_vals):
        action = sample_actions(probs, deterministic =True)
        if np.random.rand() < self.epsilon:
            action = pytorch_model.wrap(np.random.randint(self.num_outputs, size = probs.shape[0]), cuda = True)
        return action

    def set_test(self):
        self.epsilon = 0

class EpsilonStochasticProbs():

    def initialize(self, args, num_outputs):
        self.epsilon = args.greedy_epsilon
        self.num_outputs = num_outputs

    def take_action(self, probs, q_vals):
        action = sample_actions(probs, deterministic =False)
        if np.random.rand() < self.epsilon:
            action = pytorch_model.wrap(np.random.randint(self.num_outputs, size = probs.shape[0]), cuda = True)
        return action

    def set_test(self):
        self.epsilon = 0


behavior_policies = {"egq": EpsilonGreedyQ, "esq": EpsilonStochasticQ, "egp": EpsilonGreedyProbs, "esp": EpsilonStochasticProbs}