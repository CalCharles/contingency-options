from SelfBreakout.breakout_screen import Screen
from SelfBreakout.paddle import Paddle
from SelfBreakout.noblock import PaddleNoBlocks
from file_management import load_from_pickle, get_edge
import glob, os
from DopamineRL.dopamine_trainrl import train_dopamine
from DopamineRL.dopamine_models import models
from Environments.multioption import MultiOption
from ReinforcementLearning.learning_algorithms import learning_algorithms
from OptionChain.option_chain import OptionChain
from Environments.state_definition import GetState, compute_minmax
from BehaviorPolicies.behavior_policies import behavior_policies
from arguments import get_args
from ReinforcementLearning.train_rl import trainRL
from RewardFunctions.dummy_rewards import BounceReward, Xreward, BlockReward
from Pushing.screen import Pushing
import numpy as np

if __name__ == "__main__":
    # used arguments
        # record-rollouts (where data is stored for computing minmax)
        # changepoint-dir (where option chain is stored)
        # save-dir (where saved data is stored)
        # model-form
        # true-environment
        # train-edge
        # state-forms
        # state-names
    # Example usage: 
    # python paddle_bounce.py --model-form tab --optimizer-form TabQ --record-rollouts "data/action/" --train-edge "Paddle->Ball" --num-stack 1 --train --num-iters 100000 --save-dir data/paddleballtest --state-forms prox --state-names Paddle --base-node Paddle --changepoint-dir data/paddlegraph --factor 8 --greedy-epsilon .2 --lr .01 --normalize --behavior-policy egq --gamma .99 > out.txt
    # python paddle_bounce.py --model-form fourier --optimizer-form SARSA --record-rollouts "data/action/" --train-edge "Paddle->Ball" --num-stack 2 --train --num-iters 100000 --save-dir data/paddleballpg --state-forms xprox --state-names Paddle --base-node Paddle --changepoint-dir data/paddlegraphpg --factor 10 --num-layers 1 --greedy-epsilon .1 --lr .001 --normalize --behavior-policy egq --save-dir data/xstates/ --optim base > out.txt
    # python dopamine_paddle.py --record-rollouts data/integrationpaddle --changepoint-dir data/dopegraph --model-form rainbow --true-environment --train-edge "Action->Reward" --state-forms raw --state-names Action --num-steps 5 --num-stack 4 --num-iters 2000000 --log-interval 200 --save-dir ../datasets/caleb_data/dopamine/rainbow/ --optim base > baselines/rainbow.txt
    # python dopamine_paddle.py --record-rollouts data/extragripper --changepoint-dir data/dopepushgraph --model-form rainbow --true-environment --train-edge "Action->Reward" --state-forms raw --state-names Action --num-steps 5 --num-stack 4 --num-iters 10000000 --log-interval 200 --save-dir ../datasets/caleb_data/dopamine/rainbowpushing/ --optim base --env SelfPusher > pushingrainbow.txt
    # python dopamine_paddle.py --record-rollouts data/extragripper --changepoint-dir data/dopepushgraph --model-form rainbow --true-environment --train-edge "Action->Reward" --state-forms bounds bounds bounds prox prox --state-names Gripper Block Target Gripper__Block Block__Target --num-steps 5 --num-stack 1 --num-iters 10000000 --log-interval 200 --save-dir ../datasets/caleb_data/dopamine/rainbowpushing/ --optim base --env SelfPusher --gpu 3 --frameskip 3 --normalize --reward-form rawdist > pushingrainbowstate.txt
    args = get_args()
    # true_environment = Paddle()
    # true_environment = PaddleNoBlocks()
    if args.env == "SelfPusher":
        true_environment = Pushing(True, frameskip=args.frameskip)
    else:
        true_environment = Screen()
    dataset_path = args.record_rollouts
    changepoint_path = args.changepoint_dir
    option_chain = OptionChain(true_environment, args.changepoint_dir, args.train_edge, args)
    if args.reward_form == 'rawdist' and args.env == 'SelfPusher':
        true_environment.use_distance_reward()
        args.reward_form = 'raw'


    head, tail = get_edge(args.train_edge)
    
    reward_classes = [BlockReward(args)]

    if args.reward_form == 'x':
        reward_classes = [Xreward(args)]
    elif args.reward_form.find('move_dirall') != -1:
        reward_classes = [RewardDirection(args, 1), RewardDirection(args, 2), RewardDirection(args, 3), RewardDirection(args, 4)]
    else:
        reward_classes = [BounceReward(-1, args)]
    # reward_classes = [BounceReward(0, args), BounceReward(1, args), BounceReward(2, args), BounceReward(3, args)]
    train_models = MultiOption(len(reward_classes), models[args.model_form])
    environments = option_chain.initialize(args)
    proxy_environment = environments.pop(-1)
    proxy_chain = environments
    if len(environments) > 1: # there is a difference in the properties of a proxy environment and the true environment
        num_actions = len(environments[-1].reward_fns)
    else:
        num_actions = environments[-1].num_actions
    print(args.state_names, args.state_forms)
    state_class = GetState(head, state_forms=list(zip(args.state_names, args.state_forms)))
    # softcomputed minmax (buggy)
    state_class.minmax = compute_minmax(state_class, dataset_path)
    # HARDCODED MINMAX AT 84,84!!!!
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
    print("state class minmax", state_class.minmax)

    for reward_class in reward_classes:
        reward_class.traj_dim = state_class.shape
    print(state_class.minmax)
    behavior_policy = behavior_policies[args.behavior_policy]() # choice of policy is irrelevant
    # behavior_policy = EpsilonGreedyProbs() 
    train_dopamine(args, option_chain.save_dir, true_environment, train_models, proxy_environment,
            proxy_chain, reward_classes, state_class, num_actions, behavior_policy=behavior_policy)
