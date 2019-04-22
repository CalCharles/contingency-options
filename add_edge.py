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
    load_param)
from SelfBreakout.focus_screen import FocusEnvironment

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
    args = get_args()
    # loading vision model
    paddle_model_net_params_path = 'ObjectRecognition/net_params/two_layer.json'
    net_params = json.loads(open(paddle_model_net_params_path).read())
    params = load_param('ObjectRecognition/models/paddle.npy')
    paddle_model = ModelFocusCNN(
        image_shape=(84, 84),
        net_params=net_params,
    )
    paddle_model.set_parameters(params)
    ball_model_net_params_path = 'ObjectRecognition/net_params/two_layer.json'
    net_params = json.loads(open(ball_model_net_params_path).read())
    params = load_param('ObjectRecognition/models/ball.npy')
    ball_model = ModelFocusCNN(
        image_shape=(84, 84),
        net_params=net_params,
    )
    ball_model.set_parameters(params)
    model = ModelCollectionDAG()
    model.add_model('paddle', paddle_model, [])
    model.add_model('ball', ball_model, ['paddle'])
    ####

    true_environment = FocusEnvironment(model)
    dataset_path = args.record_rollouts
    changepoint_path = args.changepoint_dir
    option_chain = OptionChain(true_environment, args.changepoint_dir, args.train_edge, args)
    reward_paths = glob.glob(os.path.join(option_chain.save_dir, "*rwd.pkl"))
    print(reward_paths)
    reward_paths.sort(key=lambda x: int(x.split("_")[2]))

    head, tail = get_edge(args.train_edge)

    reward_classes = [load_from_pickle(pth) for pth in reward_paths]
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
    state_class = GetState(num_actions, head, state_forms=list(zip(args.state_names, args.state_forms)))
    state_class.minmax = compute_minmax(state_class, dataset_path)
    behavior_policy = behavior_policies[args.behavior_policy]()
    # behavior_policy = EpsilonGreedyProbs()
    trainRL(args, option_chain.save_dir, true_environment, train_models, learning_algorithm, proxy_environment,
            proxy_chain, reward_classes, state_class, behavior_policy=behavior_policy)
