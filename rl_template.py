import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import sys, glob, copy, os, collections, time
from arguments import get_args
from ReinforcementLearning.learning_algorithms import learning_algorithms
from Models.models import models 
from ReinforcementLearning.train_rl import trainRL
from Environments.environment_specification import ChainMDP, ProxyEnvironment
from Environments.state_definition import GetRaw
from Environments.multioption import MultiOption
from RewardFunctions.dummy_rewards import RewardRight, RewardCenter, RewardLeft
from BehaviorPolicies.behavior_policies import EpsilonGreedyQ, EpsilonGreedyProbs
from OptionChain.option_chain import OptionChain
import collections


if __name__ == "__main__":
    # Example command line:
        # python rl_template.py --model-form basic --optimizer-form DQN --record-rollouts "data/testchain/" --train-edge "Action->chain" --num-stack 1 --train --num-iters 10000 --save-dir data/test
        # python rl_template.py --model-form tab --optimizer-form TabQ --record-rollouts "data/testchain/" --train-edge "Action->chain" --num-stack 1 --train --num-iters 10000 --save-dir data/test
        # with 3 rewards: python rl_template.py --model-form basic --optimizer-form DQN --record-rollouts "data/testchain/" --train-edge "Action->chain" --num-stack 1 --train --num-iters 1000 --save-dir data/test --num-update-model 1
        # python rl_template.py --model-form tab --optimizer-form TabQ --record-rollouts "data/testchain/" --train-edge "Action->chain" --num-stack 1 --train --num-iters 1000 --save-dir data/test --num-update-model 1
    args = get_args()
    true_environment = ChainMDP(30)
    # train_models = MultiOption(1, BasicModel)
    reward_classes = [RewardLeft(None, args), RewardCenter(None, args), RewardRight(None, args)]
    train_models = MultiOption(len(reward_classes), models[args.model_form])
    learning_algorithm = learning_algorithms[args.optimizer_form]()
    option_chain = OptionChain(true_environment, args.record_rollouts, args.train_edge, args) # here, train_edge should act like a save folder
    minmax = (0,30)
    state_class = GetRaw(3, minmax = minmax, state_shape = [1])
    behavior_policy = EpsilonGreedyQ()
    # behavior_policy = EpsilonGreedyProbs()
    proxy_chain = option_chain.initialize(args) # the last term is None since the last environment is not yet made
    proxy_chain.pop(-1)
    print(proxy_chain)
    trainRL(args, option_chain.save_dir, true_environment, train_models, learning_algorithm, 
            proxy_chain, reward_classes, state_class, behavior_policy=behavior_policy)
