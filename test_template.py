import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import sys, glob, copy, os, collections, time
from arguments import get_args
from ReinforcementLearning.learning_algorithms import learning_algorithms
from ReinforcementLearning.models import models 
from ReinforcementLearning.test_policies import testRL
from Environments.environment_specification import ChainMDP, ProxyEnvironment
from Environments.state_definition import GetRaw
from Environments.multioption import MultiOption
from BehaviorPolicies.behavior_policies import EpsilonGreedyQ, EpsilonGreedyProbs
from OptionChain.option_chain import OptionChain
import collections


if __name__ == "__main__":
    # Example Command Line
    # python test_template.py --train-edge "Action->chain" --num-stack 1 --num-iters 1000 --save-dir data/testtest/ --record-rollouts data/testchain/ --greedy-epsilon 0
    args = get_args()
    true_environment = ChainMDP(30)
    option_chain = OptionChain(true_environment, args.record_rollouts, args.train_edge, args) # here, train_edge should act like a save folder
    minmax = (0,30)
    state_class = GetRaw(3, minmax = minmax, state_shape = [1])
    behavior_policy = EpsilonGreedyQ() # make sure to set epsilon to 0
    # behavior_policy = EpsilonGreedyProbs()
    proxy_chain = option_chain.initialize(args) # the last term not None since the last environment is loaded for testing
    print(proxy_chain)
    testRL(args, option_chain.save_dir, true_environment, proxy_chain, 
            state_class, behavior_policy=behavior_policy)
