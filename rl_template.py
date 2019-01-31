import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import sys, glob, copy, os, collections, time
from arguments import get_args
from ReinforcementLearning.learning_algorithms import TabQ_optimizer, DQN_optimizer, PPO_optimizer, A2C_optimizer, SARSA_optimizer
from ReinforcementLearning.models import models 
from Environments.environment_specification import ChainMDP, ProxyEnvironment


if __name__ == "__main__":
    args = get_args()
    true_environment = ChainMDP(100)
    # train_models = MultiOption(1, BasicModel)
    train_models = MultiOption(1, TabularQ)
    # learning_algorithm = DQN_optimizer()
    learning_algorithm = TabQ_optimizer()
    # learning_algorithm = DDPG_optimizer()
    option_chain = OptionChain(true_environment)
    reward_classes = [RewardRight()]
    state_class = GetRaw((1,), 3, true_environment.minmax)
    behavior_policy = EpsilonGreedyQ()
    # behavior_policy = EpsilonGreedyProbs()
    trainRL(args, true_environment, train_models, learning_algorithm, 
            option_chain, reward_classes, state_class, behavior_policy=behavior_policy)
