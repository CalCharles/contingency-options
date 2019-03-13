from SelfBreakout.breakout_screen import Screen
from file_management import load_from_pickle, get_edge
import glob, os
from Models.models import models
from Environments.multioption import MultiOption
from ReinforcementLearning.learning_algorithms import learning_algorithms
from OptionChain.option_chain import OptionChain
from Environments.state_definition import GetState, compute_minmax
from BehaviorPolicies.behavior_policies import EpsilonGreedyQ, EpsilonGreedyProbs
from arguments import get_args
from ReinforcementLearning.train_rl import trainRL

if __name__ == "__main__":
    # used arguments
        # record-rollouts (where data is stored)
        # changepoint-dir (where option chain is stored)
        # model-form
        # optimizer-form
        # train-edge
        # state-forms
        # state-names

    args = get_args()
    true_environment = Ball()
    dataset_path = args.record_rollouts
    changepoint_path = args.changepoint_dir
    option_chain = OptionChain(true_environment, args.changepoint_dir, args.train_edge, args)

    head, tail = get_edge(args.train_edge)

    reward_classes = [block_rewards()]
    # reward_classes = [bounce_rewards(0), bounce_rewards(1), bounce_rewards(2), bounce_rewards(3)]
    train_models = MultiOption(len(reward_paths), models[args.model_form])
    learning_algorithm = learning_algorithms[args.optimizer_form]()
    environments = option_chain.initialize(args)
    environments.pop(-1)
    proxy_chain = environments
    if len(environments) > 1: # there is a difference in the properties of a proxy environment and the true environment
        num_actions = len(environments[-1].reward_fns)
    else:
        num_actions = environments[-1].num_actions
    print(args.state_names, args.state_forms)
    state_class = GetState(num_actions, tail, state_forms=list(zip(args.state_names, args.state_forms)))
    state_class.minmax = compute_minmax(state_class, dataset_path)
    print(state_class.minmax)
    behavior_policy = EpsilonGreedyQ()
    # behavior_policy = EpsilonGreedyProbs()
    trainRL(args, option_chain.save_dir, true_environment, train_models, learning_algorithm, 
            proxy_chain, reward_classes, state_class, behavior_policy=behavior_policy)
