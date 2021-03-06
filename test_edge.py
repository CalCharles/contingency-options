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
from ReinforcementLearning.test_policies import testRL
from Environments.environment_specification import ChainMDP, ProxyEnvironment
from Environments.state_definition import GetRaw, GetState, compute_minmax
from Environments.multioption import MultiOption
from Pushing.screen import Pushing
from SelfBreakout.paddle import Paddle
from RewardFunctions.changepointReward import ChangepointMarkovReward
from RewardFunctions.dummy_rewards import BounceReward, RewardDirection, RawReward
from BehaviorPolicies.behavior_policies import behavior_policies
from OptionChain.option_chain import OptionChain
from file_management import get_edge, load_from_pickle
import collections, json
from ObjectRecognition.model import (
    ModelFocusCNN, ModelCollectionDAG,
    load_param, util)
from SelfBreakout.focus_screen import FocusEnvironment
from AtariEnvironments.focus_atari import FocusAtariEnvironment



if __name__ == "__main__":
    # Example Command Line
    # python test_template.py --train-edge "Action->chain" --num-stack 1 --num-iters 1000 --changepoint-dir data/optgraph --save-dir data/testtest/ --record-rollouts data/testchain/ --greedy-epsilon 0
    # python test_edge.py --model-form population --optimizer-form CMAES --record-rollouts "data/action/" --train-edge "Paddle->Ball" --num-stack 1 --num-frames 10000 --state-forms prox svel bounds --state-names Paddle Ball Ball --base-node Paddle --changepoint-dir ./data/paddlegraph --behavior-policy esp --reward-form bounce --gamma .9 --init-form xnorm --num-layers 1 --select-ratio .2 --num-population 10 --sample-duration 100 --warm-up 0 --log-interval 1 --scale 1 --gpu 2
    # python test_edge.py --model-form population --optimizer-form CMAES --record-rollouts "data/action/" --train-edge "Paddle->Ball" --num-stack 1 --state-forms prox svel bounds --state-names Paddle Ball Ball --base-node Paddle --changepoint-dir ./data/paddlegraph2 --behavior-policy esp --reward-form dir --gamma .9 --init-form xnorm --num-layers 1 --select-ratio .2 --num-population 10 --sample-duration 100 --warm-up 0 --log-interval 1 --scale 1 --gpu 1 --load-weights --num-iters 1000
    # python test_edge.py --train-edge "Action->Gripper" --num-stack 1 --num-iters 2500 --changepoint-dir data/pushergraph --frameskip 3 --num-stack 2 --state-forms bounds --state-names Gripper --save-dir data/extragripper/ --record-rollouts data/testchain/ --greedy-epsilon 0 --num-update-model 2 --true-environment --env SelfPusher --behavior-policy egq --load-weights
    # python test_edge.py --train-edge "Gripper->Block" --num-stack 1 --num-iters 500 --changepoint-dir data/grippergraph2 --frameskip 3 --state-forms prox bounds bounds --state-names Gripper Gripper Block --save-dir data/extragripper/ --record-rollouts data/testchain/ --greedy-epsilon 0 --num-update-model 30 --true-environment --env SelfPusher --behavior-policy esp --load-weights --reward-form move_dirall --base-node Gripper
    args = get_args()
    torch.cuda.set_device(args.gpu)
    # # loading vision model
    paddle_model_net_params_path = 'ObjectRecognition/net_params/attn_base.json'
    # paddle_model_net_params_path = 'ObjectRecognition/net_params/attn_softmax.json'
    net_params = json.loads(open(paddle_model_net_params_path).read())
    params = load_param('results/cmaes_soln/focus_self/paddle.pth')
    # params = load_param('ObjectRecognition/models/atari/paddle_bin_smooth.pth')
    paddle_model = ModelFocusCNN(
        image_shape=(84, 84),
        net_params=net_params,
        binarize = 0.000
    )
    paddle_model.set_parameters(params)
    # ball_model_net_params_path = 'ObjectRecognition/net_params/attn_base.json'
    ball_model_net_params_path = 'ObjectRecognition/net_params/attn_softmax.json'
    net_params = json.loads(open(ball_model_net_params_path).read())
    params = load_param('results/cmaes_soln/focus_self/ball.pth')
    # params = load_param('ObjectRecognition/models/atari/42531_2_smooth_3_2.pth')
    ball_model = ModelFocusCNN(
        image_shape=(84, 84),
        net_params=net_params,
        binarize = 0.0
    )
    ball_model.set_parameters(params)
    model = ModelCollectionDAG()
    model.add_model('Paddle', paddle_model, [], augment_fn=util.RemoveMeanMemory(nb_size=(8, 8)))
    f1 = util.LowIntensityFiltering(6.0)
    f2 = util.JumpFiltering(3, 0.05)
    def f(x, y):
        return f2(x, f1(x, y))
        # model.add_model('train', r_model, ['premise'], augment_pt=f)

    model.add_model('Ball', ball_model, ['Paddle'], augment_pt=f)#,augment_pt=util.JumpFiltering(2, 0.05))
    # ####

    if args.true_environment:
        model = None
    print(args.true_environment, args.env)
    if args.env == 'SelfPusher':
        if args.true_environment:
            true_environment = Pushing(pushgripper=True, frameskip=args.frameskip)
        else:
            true_environment = None # TODO: implement
    elif args.env == 'SelfBreakout':
        if args.true_environment:
            true_environment = Screen(frameskip=args.frameskip)
        else:
            true_environment = FocusEnvironment(model, display=args.display_focus)
    elif args.reward_form in ['bounce', 'dir', 'dirneg']:
        true_environment = Paddle()
    elif args.env.find('Atari') != -1:
        true_environment = FocusAtariEnvironment(model, args.env[len("Atari"):], args.seed, 0, args.save_dir)
    dataset_path = args.record_rollouts
    changepoint_path = args.changepoint_dir
    option_chain = OptionChain(true_environment, args.changepoint_dir, args.train_edge, args)


    head, tail = get_edge(args.train_edge)
    if args.reward_form == 'raw':
        reward_classes = [RawReward(args)]
    elif args.reward_form.find('move_dirall') != -1:
        reward_classes = [RewardDirection(args, 1), RewardDirection(args, 2), RewardDirection(args, 3), RewardDirection(args, 4)]
    elif args.reward_form.find('dir') != -1:
        reward_classes = [BounceReward(0, args), BounceReward(1, args), BounceReward(2, args), BounceReward(3, args)]
    elif args.reward_form == 'bounce':
        reward_classes = [BounceReward(-1, args)]
    else:
        reward_paths = glob.glob(os.path.join(option_chain.save_dir, "*rwd.pkl"))
        if len(reward_paths) > 0:
            print(reward_paths)
            reward_paths.sort(key=lambda x: int(x.split("__")[2]))
            reward_classes = [load_from_pickle(pth) for pth in reward_paths]
            for rc in reward_classes:
                if type(rc) == ChangepointMarkovReward:
                    rc.markovModel = rc.markovModel.cuda(args.gpu)
        else:
            reward_classes = None

    environments = option_chain.initialize(args)
    proxy_environment = environments.pop(-1)
    proxy_chain = environments
    if args.load_weights:
        train_models = proxy_environment.models
    if len(environments) > 1: # there is a difference in the properties of a proxy environment and the true environment
        num_actions = len(environments[-1].reward_fns)
    else:
        num_actions = environments[-1].num_actions
    print(num_actions, true_environment)
    print(args.state_names, args.state_forms)
    state_class = GetState(head, state_forms=list(zip(args.state_names, args.state_forms)))
    if args.normalize:
        state_class.minmax = compute_minmax(state_class, dataset_path)
    print(state_class.minmax)
    behavior_policy = behavior_policies[args.behavior_policy]()
    # behavior_policy = EpsilonGreedyProbs()
    testRL(args, option_chain.save_dir, true_environment, proxy_chain, proxy_environment,
            state_class, behavior_policy=behavior_policy, num_actions= num_actions, reward_classes = reward_classes)
