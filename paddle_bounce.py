from SelfBreakout.breakout_screen import Screen
from SelfBreakout.paddle import Paddle
from SelfBreakout.noblock import PaddleNoBlocks
from SelfBreakout.unobstructpaddle import PaddleNoWalls
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
from RewardFunctions.dummy_rewards import BounceReward, Xreward
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
    # python paddle_bounce.py --model-form tab --optimizer-form TabQ --record-rollouts "data/action/" --train-edge "Paddle->Ball" --num-stack 1 --train --num-iters 100000 --save-dir data/paddleballtest --state-forms prox --state-names Paddle --base-node Paddle --changepoint-dir data/paddlegraph --factor 8 --greedy-epsilon .2 --lr .01 --normalize --behavior-policy egq --gamma .99 > out.txt
    # python paddle_bounce.py --model-form fourier --optimizer-form SARSA --record-rollouts "data/action/" --train-edge "Paddle->Ball" --num-stack 2 --train --num-iters 100000 --save-dir data/paddleballpg --state-forms xprox --state-names Paddle --base-node Paddle --changepoint-dir data/paddlegraphpg --factor 10 --num-layers 1 --greedy-epsilon .1 --lr .001 --normalize --behavior-policy egq --save-dir data/xstates/ --optim base > out.txt
    # x values python paddle_bounce.py --model-form basic --optimizer-form PPO --record-rollouts "data/action/" --train-edge "Paddle->Ball" --num-stack 2 --train --num-iters 10000 --state-forms xprox --state-names Paddle --base-node Paddle --changepoint-dir ../datasets/caleb_data/cotest/paddlegraph --factor 16 --num-layers 2 --lr 7e-4 --behavior-policy esp --optim RMSprop --period .05 --reward-form x --gamma .5 --save-dir data/doperl --init-form xnorm --entropy-coef 0.01 --grad-epoch 5
    # x values bounce 100k python paddle_bounce.py --model-form gaumulti --optimizer-form PPO --record-rollouts "data/action/" --train-edge "Paddle->Ball" --num-stack 2 --train --num-iters 100000 --state-forms xprox --state-names Paddle --base-node Paddle --changepoint-dir ../datasets/caleb_data/cotest/paddlegraph --factor 40 --num-layers 1 --lr 7e-4 --behavior-policy esp --optim RMSprop --period .01 --scale 30 --num-population 50 --normalize --reward-form bounce --gamma .99 --save-dir data/doperl --init-form xnorm --entropy-coef 0.1 --grad-epoch 5 > outdope.txt
    # evo python paddle_bounce.py --model-form population --optimizer-form Evo --record-rollouts "data/action/" --train-edge "Paddle->Ball" --num-stack 1 --train --num-iters 20000 --state-forms prox --state-names Paddle --base-node Paddle --changepoint-dir ../datasets/caleb_data/cotest/paddlegraph --lr 5e-2 --behavior-policy esp --reward-form bounce --gamma .8 --init-form xnorm --factor 4 --num-layers 2 --evolve-form basic --select-ratio .25 --num-population 20 --sample-duration 100 --sample-schedule 8 --elitism --warm-up 0 --log-interval 1 --scale 5
    args = get_args()
    # true_environment = Paddle()
    true_environment = PaddleNoWalls()
    # true_environment = PaddleNoBlocks()
    dataset_path = args.record_rollouts
    changepoint_path = args.changepoint_dir
    option_chain = OptionChain(true_environment, args.changepoint_dir, args.train_edge, args)


    head, tail = get_edge(args.train_edge)

    if args.reward_form == 'x':
        reward_classes = [Xreward(args)]
    elif args.reward_form == 'bounce':
        reward_classes = [BounceReward(-1, args)]
    elif args.reward_form == 'dir':
        reward_classes = [BounceReward(0, args), BounceReward(1, args), BounceReward(2, args), BounceReward(3, args)]
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
    state_class = GetState(num_actions, head, state_forms=list(zip(args.state_names, args.state_forms)))
    if args.normalize:
        state_class.minmax = compute_minmax(state_class, dataset_path)
        print(state_class.minmax)
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
    trainRL(args, option_chain.save_dir, true_environment, train_models, learning_algorithm, proxy_environment,
            proxy_chain, reward_classes, state_class, behavior_policy=behavior_policy)
