from Pushing.screen import Pushing
from file_management import load_from_pickle, get_edge
import glob, os, torch
import numpy as np
from Models.models import models
from Environments.multioption import MultiOption
from ReinforcementLearning.learning_algorithms import learning_algorithms
from OptionChain.option_chain import OptionChain
from Environments.state_definition import GetState, compute_minmax
from BehaviorPolicies.behavior_policies import behavior_policies
from arguments import get_args
from ReinforcementLearning.train_rl import trainRL
from RewardFunctions.dummy_rewards import BounceReward, Xreward, BlockReward, RewardDirection, RawReward
from RewardFunctions.changepointReward import compute_cp_minmax
from RewardFunctions.novelty_wrappers import novelty_rewards

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
    # python block_move.py --model-form vector --optimizer-form DQN --record-rollouts "data/ballpush/" --train-edge "Gripper->Block" --num-stack 1 --train --num-iters 100 --state-forms prox bounds bounds --state-names Block Gripper Block --base-node Gripper --changepoint-dir ./data/grippergraph --lr 7e-4 --behavior-policy egq --reward-form block --gamma .99 --init-form smalluni --num-layers 1 --reward-check 10 --greedy-epsilon .03 --reward-form move_dirall --log-interval 10 --num-steps 5
    args = get_args()
    torch.cuda.set_device(args.gpu)
    true_environment = Pushing(True, frameskip=args.frameskip)
    # true_environment = PaddleNoWalls()
    # true_environment = PaddleNoBlocks()
    dataset_path = args.record_rollouts
    changepoint_path = args.changepoint_dir
    option_chain = OptionChain(true_environment, args.changepoint_dir, args.train_edge, args)


    head, tail = get_edge(args.train_edge)
    if args.reward_form.find('move_dirtouch') != -1:
        reward_classes = [RewardDirection(args, -1)]
    elif args.reward_form.find('move_dirall') != -1:
        reward_classes = [RewardDirection(args, 1), RewardDirection(args, 2), RewardDirection(args, 3), RewardDirection(args, 4)]
    elif args.reward_form.find('move_dir') != -1:
        direc = int(args.reward_form[-1])
        reward_classes = [RewardDirection(args, direc)]
    elif args.reward_form.find('raw') != -1:
        reward_classes = [RawReward(args)]


    train_models = MultiOption(len(reward_classes), models[args.model_form])
    learning_algorithm = learning_algorithms[args.optimizer_form]()
    environments = option_chain.initialize(args)
    proxy_environment = environments.pop(-1)
    proxy_chain = environments
    if args.load_weights:
        train_models = proxy_environment.models
    if len(environments) > 1: # there is a difference in the properties of a proxy environment and the true environment
        num_actions = len(environments[-1].reward_fns)
    else:
        num_actions = environments[-1].num_actions
    print(args.state_names, args.state_forms)
    state_class = GetState(head, state_forms=list(zip(args.state_names, args.state_forms)))
    if args.normalize:
        minv = []
        maxv = []
        for f in args.state_forms:
            if f == 'prox':
                minv += [-84,-84]
                maxv += [84,84]
            elif f == 'bounds':
                minv += [0,0]
                maxv += [84,84]
        state_class.minmax = np.stack((np.array(minv), np.array(maxv)))
        print(state_class.minmax)

        # state_class.minmax = compute_minmax(state_class, dataset_path)
        # print(state_class.minmax)
    new_reward_classes = []
    cp_minmax = compute_cp_minmax(reward_classes[0], dataset_path)
    for reward_class in reward_classes:
        reward = reward_class
        for wrapper in args.novelty_wrappers:
            reward = novelty_rewards[wrapper](args, reward, minmax = cp_minmax)
        new_reward_classes.append(reward)
    reward_classes = new_reward_classes
    behavior_policy = behavior_policies[args.behavior_policy]()
    # behavior_policy = EpsilonGreedyProbs()
    save_graph = args.save_graph
    if args.save_dir == "graph":
        save_graph = option_chain.save_dir

    trainRL(args, save_graph, true_environment, train_models, learning_algorithm, proxy_environment,
            proxy_chain, reward_classes, state_class, behavior_policy=behavior_policy)
