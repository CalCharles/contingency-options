from SelfBreakout.breakout_screen import Screen
from file_management import load_from_pickle, get_edge
import glob, os
from Models.models import models
from Environments.multioption import MultiOption
from ReinforcementLearning.learning_algorithms import learning_algorithms
from OptionChain.option_chain import OptionChain
from Environments.state_definition import GetState, compute_minmax
from BehaviorPolicies.behavior_policies import behavior_policies
from arguments import get_args
from ReinforcementLearning.train_rl import trainRL
from RewardFunctions.changepointReward import ChangepointMarkovReward
from ObjectRecognition.model import (
    ModelFocusCNN, ModelCollectionDAG,
    load_param, util)
from SelfBreakout.focus_screen import FocusEnvironment
import json
import torch
from AtariEnvironments.focus_atari import FocusAtariEnvironment
from RewardFunctions.dummy_rewards import RawReward

if __name__ == "__main__":
    # used arguments
        # record-rollouts (where data is stored)
        # changepoint-dir (where option chain is stored)
        # model-form
        # optimizer-form
        # train-edge
        # state-forms
        # state-names
    # Usage Example:
        # add Action->Paddle: python add_edge.py --model-form basic --optimizer-form DQN --record-rollouts "data/random/" --train-edge "Action->Paddle" --num-stack 2 --train --num-iters 10000 --save-dir data/action --state-forms bounds --state-names Paddle
        # Using tabular Action->Paddle:  python add_edge.py --model-form tab --optimizer-form TabQ --record-rollouts "data/random/" --train-edge "Action->Paddle" --num-stack 1 --train --num-iters 10000 --save-dir data/action --state-forms bounds --state-names Paddle --num-update-model 1
        # Action->Paddle: python add_edge.py --model-form basic --optimizer-form DQN --record-rollouts "data/random/" --train-edge "Action->Paddle" --changepoint-dir data/integrationgraph --num-stack 2 --factor 6 --train --num-iters 1000 --save-dir data/action --state-forms bounds --state-names Paddle --num-steps 1 --reward-check 5 --num-update-model 1 --greedy-epsilon .1 --lr 1e-2 --init-form smalluni --behavior-policy egq --grad-epoch 5 --entropy-coef .01 --value-loss-coef 0.5 --gamma .9 --save-models --save-dir data/integrationpaddle --save-graph data/intnetpaddle > integration/paddle.txt
        # python add_edge.py --model-form population --optimizer-form CMAES --record-rollouts "data/integrationpaddle/" --train-edge "Paddle->Ball" --num-stack 1 --train --num-iters 30 --state-forms prox vel --state-names Paddle Ball --changepoint-dir ./data/integrationgraph/ --lr 5e-3 --behavior-policy esp --reward-form bounce --gamma .87 --init-form xuni --factor 8 --num-layers 1 --base-form basic --select-ratio .2 --num-population 10 --sample-duration 100 --sample-schedule 12 --warm-up 0 --log-interval 1 --scale 2 --reward-check 10 --save-models --save-dir data/integrationbounce > integration/ball.txt
        # atari python add_edge.py --model-form basic --optimizer-form DQN --record-rollouts "data/atarirandom/" --train-edge "Action->Paddle" --changepoint-dir data/atarigraph/ --num-stack 2 --factor 6 --train --num-iters 1000 --save-dir data/action --state-forms bounds --state-names Paddle --num-steps 1 --reward-check 3 --changepoint-queue-len 10 --num-update-model 1 --greedy-epsilon .1 --lr 1e-2 --init-form smalluni --behavior-policy egq --grad-epoch 5 --entropy-coef .01 --value-loss-coef 0.5 --gamma 0.1 --focus-dumps-name focus_dumps.txt --env AtariBreakoutNoFrameskip-v0 --save-models --save-dir data/ataripaddle --save-graph data/atarinetpaddle > atari/paddle.txt
        # python add_edge.py --model-form population --optimizer-form CMAES --record-rollouts "data/integrationpaddle/" --train-edge "Paddle->Ball" --num-stack 1 --train --num-iters 30 --state-forms prox vel --state-names Paddle Ball --changepoint-dir ./data/integrationgraph/ --lr 5e-3 --behavior-policy esp --reward-form bounce --gamma .87 --init-form xuni --factor 8 --num-layers 1 --base-form basic --select-ratio .2 --num-population 10 --sample-duration 100 --sample-schedule 15 --warm-up 0 --log-interval 1 --scale 2 --reward-check 10 --focus-dumps-name focus_dumps.txt --env AtariBreakoutNoFrameskip-v0 --save-models --save-dir data/ataribounce  > atari/ball.txt
        # first train: python add_edge.py --model-form population --optimizer-form CMAES --record-rollouts "data/integrationpaddle/" --train-edge "Paddle->Ball" --num-stack 1 --train --num-iters 100 --state-forms prox vel vel --state-names Paddle Ball Paddle --changepoint-dir ./data/atarigraph/ --lr 5e-3 --greedy-epsilon .01 --behavior-policy esp --gamma 0 --init-form smalluni --factor 12 --num-layers 1 --base-form basic --num-population 10 --retest 2 --OoO-eval --sample-duration 100 --sample-schedule 15 --done-swapping 0 --warm-up 0 --log-interval 1 --init-var 5e-2 --scale 1 --reward-check 20 --focus-dumps-name focus_dumps.txt --env AtariBreakoutNoFrameskip-v0 --save-dir data/atariball --save-models --save-graph data/atariballgraph --save-interval 1  > atariball.txt
        # train baseline: python add_edge.py --model-form raw --optimizer-form A2C --record-rollouts "data/random/" --train-edge "Action->Reward" --num-stack 4 --train --num-iters 1000000 --state-forms raw --state-names Paddle --changepoint-dir ./data/rawgraph/ --reward-form raw --lr 7e-4 --greedy-epsilon 0 --value-loss-coef 0.5 --optim RMSprop --behavior-policy esp --gamma 0.99 --init-form orth --factor 16 --num-layers 1 --warm-up 0 --log-interval 100 --entropy-coef .01 --normalize --reward-check 5 --changepoint-queue 5 --env AtariBreakoutNoFrameskip-v0 --gpu 3 --true-environment --lag-num 0 --post-transform-form linear --return-form value > a2c.txt
        # python add_edge.py --model-form raw --optimizer-form PPO --record-rollouts "data/random/" --train-edge "Action->Reward" --num-stack 4 --train --num-iters 1000000 --state-forms raw --state-names Paddle --changepoint-dir ./data/rawgraph/ --reward-form raw --lr 2.5e-4 --greedy-epsilon 0 --gamma 0.99 --value-loss-coef 0.5 --optim RMSprop --init-form orth --factor 16 --num-layers 1 --warm-up 0 --log-interval 10 --entropy-coef .01 --normalize --reward-check 128 --changepoint-queue 128 --buffer-clip 128 --num-grad-states 32 --grad-epoch 4 --clip-param 0.1 --env AtariBreakoutNoFrameskip-v0 --gpu 2 --true-environment --lag-num 0 --post-transform-form linear --return-form normal > ataribaseline.txt
        # 
    args = get_args()
    torch.cuda.set_device(args.gpu)

    # loading vision model
    paddle_model_net_params_path = 'ObjectRecognition/net_params/attn_base.json'
    # paddle_model_net_params_path = 'ObjectRecognition/net_params/attn_softmax.json'
    net_params = json.loads(open(paddle_model_net_params_path).read())
    params = load_param('results/cmaes_soln/focus_self/paddle.pth')
    # params = load_param('ObjectRecognition/models/atari/paddle_bin_smooth.pth')
    paddle_model = ModelFocusCNN(
        image_shape=(84, 84),
        net_params=net_params,
        binarize = 0.001
    )
    paddle_model.set_parameters(params)
    # ball_model_net_params_path = 'ObjectRecognition/net_params/attn_base.json'
    ball_model_net_params_path = 'ObjectRecognition/net_params/attn_softmax.json'
    net_params = json.loads(open(ball_model_net_params_path).read())
    # params = load_param('ObjectRecognition/models/self/ball_bin_long_smooth.pth')
    params = load_param('ObjectRecognition/models/atari/42531_2_smooth_3_2.pth')
    ball_model = ModelFocusCNN(
        image_shape=(84, 84),
        net_params=net_params,
        binarize = 0.1
    )
    ball_model.set_parameters(params)
    model = ModelCollectionDAG()
    model.add_model('Paddle', paddle_model, [], augment_fn=util.RemoveMeanMemory(nb_size=(3, 8)))
    f1 = util.LowIntensityFiltering(8.0)
    f2 = util.JumpFiltering(3, 0.05)
    def f(x, y):
        return f2(x, f1(x, y))
        # model.add_model('train', r_model, ['premise'], augment_pt=f)

    model.add_model('Ball', ball_model, ['Paddle'])#, augment_pt=f)#,augment_pt=util.JumpFiltering(2, 0.05))
    ####
    if args.true_environment:
        model = None
    print(args.true_environment, args.env)
    if args.env == 'SelfBreakout':
        if args.true_environment:
            true_environment = Screen(frameskip=args.frameskip)
        else:
            true_environment = FocusEnvironment(model)
    elif args.env.find('Atari') != -1:
        true_environment = FocusAtariEnvironment(model, args.env[len("Atari"):], args.seed, 0, args.save_dir)
    dataset_path = args.record_rollouts
    changepoint_path = args.changepoint_dir
    option_chain = OptionChain(true_environment, args.changepoint_dir, args.train_edge, args)
    reward_paths = glob.glob(os.path.join(option_chain.save_dir, "*rwd.pkl"))
    print(reward_paths)
    reward_paths.sort(key=lambda x: int(x.split("__")[2]))

    head, tail = get_edge(args.train_edge)

    if args.reward_form != 'raw':
        reward_classes = [load_from_pickle(pth) for pth in reward_paths]
        for rc in reward_classes:
            if type(rc) == ChangepointMarkovReward:
                rc.markovModel.cuda()
    else:
        reward_classes = [RawReward(args)]
    # train_models = MultiOption(1, BasicModel)
    train_models = MultiOption(len(reward_paths), models[args.model_form])
    # learning_algorithm = DQN_optimizer()
    learning_algorithm = learning_algorithms[args.optimizer_form]()
    # learning_algorithm = DDPG_optimizer()
    environments = option_chain.initialize(args)
    print(environments)
    proxy_environment = environments.pop(-1)
    proxy_chain = environments
    if len(environments) > 1: # there is a difference in the properties of a proxy environment and the true environment
        num_actions = len(environments[-1].reward_fns)
    else:
        num_actions = environments[-1].num_actions
    print(args.state_names, args.state_forms)
    state_class = GetState(head, state_forms=list(zip(args.state_names, args.state_forms)))
    state_class.minmax = compute_minmax(state_class, dataset_path, filename=args.focus_dumps_name)
    behavior_policy = behavior_policies[args.behavior_policy]()
    # behavior_policy = EpsilonGreedyProbs()
    save_graph = args.save_graph
    if args.save_dir == "graph":
        save_graph = option_chain.save_dir
    trainRL(args, save_graph, true_environment, train_models, learning_algorithm, proxy_environment,
            proxy_chain, reward_classes, state_class, behavior_policy=behavior_policy)
