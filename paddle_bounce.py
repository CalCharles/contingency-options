from SelfBreakout.breakout_screen import Screen
from SelfBreakout.paddle import Paddle
from file_management import load_from_pickle, get_edge
import glob, os
from ReinforcementLearning.models import models
from Environments.multioption import MultiOption
from ReinforcementLearning.learning_algorithms import learning_algorithms
from OptionChain.option_chain import OptionChain
from Environments.state_definition import GetState, compute_minmax
from BehaviorPolicies.behavior_policies import behavior_policies
from arguments import get_args
from ReinforcementLearning.train_rl import trainRL
from RewardFunctions.dummy_rewards import BounceReward, Xreward

if __name__ == "__main__":
    # used arguments
        # record-rollouts (where data is stored for computing minmax)
        # changepoint-dir (where option chain is stored)
        # save-dir (where saved data is stored)
        # model-form
        # optimizer-form
        # train-edge
        # state-forms
        # state-names
    # Example usage: 
    # python paddle_bounce.py --model-form tab --optimizer-form TabQ --record-rollouts "data/action/" --train-edge "Paddle->Ball" --num-stack 1 --train --num-iters 100000 --save-dir data/paddleballtest --state-forms prox --state-names Paddle --base-node Paddle --changepoint-dir data/paddlegraph --factor 8 --greedy-epsilon .2 --lr .01 --normalize --behavior-policy egq --gamma .99 > out.txt

    args = get_args()
    true_environment = Paddle()
    dataset_path = args.record_rollouts
    changepoint_path = args.changepoint_dir
    option_chain = OptionChain(true_environment, args.changepoint_dir, args.train_edge, args)


    head, tail = get_edge(args.train_edge)

    reward_classes = [BounceReward(-1, args)]
    # reward_classes = [Xreward(args)]
    # reward_classes = [BounceReward(0, args), BounceReward(1, args), BounceReward(2, args), BounceReward(3, args)]
    train_models = MultiOption(len(reward_classes), models[args.model_form])
    learning_algorithm = learning_algorithms[args.optimizer_form]()
    environments = option_chain.initialize(args)
    environments.pop(-1)
    proxy_chain = environments
    if len(environments) > 1: # there is a difference in the properties of a proxy environment and the true environment
        num_actions = len(environments[-1].reward_fns)
    else:
        num_actions = environments[-1].num_actions
    print(args.state_names, args.state_forms)
    state_class = GetState(num_actions, head, state_forms=list(zip(args.state_names, args.state_forms)))
    state_class.minmax = compute_minmax(state_class, dataset_path)
    for reward_class in reward_classes:
        reward_class.traj_dim = state_class.shape
    print(state_class.minmax)
    behavior_policy = behavior_policies[args.behavior_policy]()
    # behavior_policy = EpsilonGreedyProbs()
    trainRL(args, option_chain.save_dir, true_environment, train_models, learning_algorithm, 
            proxy_chain, reward_classes, state_class, behavior_policy=behavior_policy)
