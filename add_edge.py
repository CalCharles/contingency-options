from SelfBreakout.breakout_screen import Screen
from file_management import load_from_pickle, get_edge
import glob, os
from ReinforcementLearning.models import models
from Environments.multioption import MultiOption
from ReinforcementLearning.learning_algorithms import learning_algorithms
from OptionChain.option_chain import OptionChain
from Environments.state_definition import GetState
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
    true_environment = Screen()
    dataset_path = args.record_rollouts
    changepoint_path = args.changepoint_dir
    reward_paths = glob.glob(os.path.join(changepoint_path, args.train_edge, "*.pkl"))
    reward_paths.sort(key=lambda x: int(x.split("_")[2]))

    head, tail = get_edge(args.train_edge)

    reward_classes = [load_from_pickle(pth) for pth in reward_paths]
    # train_models = MultiOption(1, BasicModel)
    train_models = MultiOption(1, models[args.model_form])
    # learning_algorithm = DQN_optimizer()
    learning_algorithm = learning_algorithms[args.optimizer_form]()
    # learning_algorithm = DDPG_optimizer()
    option_chain = OptionChain(true_environment, args.changepoint_dir, args.train_edge)
    environments = option_chain.initialize(args)
    if len(environments) > 1: # there is a difference in the properties of a proxy environment and the true environment
        num_actions = len(environments[-1].reward_fns)
    else:
        num_actions = environments[-1].num_actions
    print(args.state_names, args.state_forms)
    state_class = GetState(num_actions, list(zip(args.state_names, args.state_forms)), head)
    behavior_policy = EpsilonGreedyQ()
    # behavior_policy = EpsilonGreedyProbs()
    trainRL(args, true_environment, train_models, learning_algorithm, 
            option_chain, reward_classes, state_class, behavior_policy=behavior_policy)
