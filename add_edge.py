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
from ObjectRecognition.model import (
    ModelFocusCNN, ModelCollectionDAG,
    load_param, util)
from SelfBreakout.focus_screen import FocusEnvironment
import json
import torch
from AtariEnvironments.focus_atari import FocusAtariEnvironment

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
        # python add_edge.py --model-form population --optimizer-form CMAES --record-rollouts "data/integrationpaddle/" --train-edge "Paddle->Ball" --num-stack 1 --train --num-iters 30 --state-forms prox vel --state-names Paddle Ball --changepoint-dir ./data/integrationgraph/ --lr 5e-3 --behavior-policy esp --reward-form bounce --gamma .87 --init-form xuni --factor 8 --num-layers 1 --base-form basic --select-ratio .2 --num-population 10 --sample-duration 100 --sample-schedule 15 --warm-up 0 --log-interval 1 --scale 2 --reward-check 10 --focus-dumps-name focus_dumps.txt --env AtariBreakoutNoFrameskip-v0 --save-models --save-dir data/ataribounce  --save-dir ../datasets/caleb_data/integrationbounce > atari/ball.txt
    args = get_args()
    torch.cuda.set_device(args.gpu)

    # loading vision model
    # paddle_model_net_params_path = 'ObjectRecognition/net_params/attn_base.json'
    paddle_model_net_params_path = 'ObjectRecognition/net_params/attn_softmax.json'
    net_params = json.loads(open(paddle_model_net_params_path).read())
    params = load_param('ObjectRecognition/models/atari/paddle_bin_smooth.pth')
    paddle_model = ModelFocusCNN(
        image_shape=(84, 84),
        net_params=net_params,
        binarize = 0.001
    )
    paddle_model.set_parameters(params)
    ball_model_net_params_path = 'ObjectRecognition/net_params/attn_softmax.json'
    net_params = json.loads(open(ball_model_net_params_path).read())
    params = load_param('ObjectRecognition/models/atari/42531_2_smooth.pth')
    ball_model = ModelFocusCNN(
        image_shape=(84, 84),
        net_params=net_params,
        binarize = 0.1
    )
    ball_model.set_parameters(params)
    model = ModelCollectionDAG()
    model.add_model('Paddle', paddle_model, [], augment_fn=util.RemoveMeanMemory(nb_size=(15, 15)), augment_pt=util.JumpFiltering(2, 0.1))
    model.add_model('Ball', ball_model, ['Paddle'], augment_pt=util.JumpFiltering(5, 0.1))
    ####

    if args.env == 'SelfBreakout':
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

    reward_classes = [load_from_pickle(pth) for pth in reward_paths]
    for rc in reward_classes:
        rc.max_dev = 1.0
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
