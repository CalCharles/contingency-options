from SelfBreakout.paddle import Paddle
from SelfBreakout.noblock import PaddleNoBlocks
from arguments import get_args
import numpy as np
from Models.pretrain_models import random_actions, action_criteria, pretrain, range_Qvals,\
                                 Q_criteria, get_states, get_option_actions, get_option_rewards, \
                                 generate_trace_training, generate_soft_dataset, supervised_criteria, \
                                 get_option_rewards, generate_distilled_training, generate_target_training
from RewardFunctions.dummy_rewards import BounceReward, Xreward, BlockReward




if __name__ == "__main__":
    '''
    record rollouts
    changepoint dir
    learning rate
    eps
    betas 
    weight decay
    save interval
    save models
    train edge
    state names
    state forms
    reward_form
    trace len
    pretrain_target
    save-dir
    '''
    # python pretrain_trace.py --model-form paramcont --record-rollouts "./data/bounce/" --train-edge "Ball->Block" --num-stack 1 --train --num-iters 1000 --state-forms bounds vel bin --state-names Ball Ball Block --base-node Paddle --changepoint-dir ./data/paddlegraph/ --factor 16 --num-layers 2 --lr 1e-5 --init-form xnorm --optimizer-form PPO --parameterized-form basic --reward-form block --pretrain-target 2 --weighting-lambda 0 --log-interval 200 --num-frames 10000 --num-grad-states 1000 --trace-len 10 --parameterized-option 2
    args = get_args()
    if args.reward_form == 'x':
        reward_classes = [Xreward(args)]
    elif args.reward_form == 'bounce':
        reward_classes = [BounceReward(-1, args)]
    elif args.reward_form == 'dir':
        reward_classes = [BounceReward(0, args), BounceReward(1, args), BounceReward(2, args), BounceReward(3, args)]
    elif args.reward_form == 'block':
        reward_classes = [BlockReward(args)]
    true_environment = Paddle()
    if args.pretrain_target == 0:
        states, resps, num_actions, state_class, proxy_chain, raws, dumps = get_states(args, true_environment, length_constraint = args.num_frames)
        actions = get_option_actions(args.record_rollouts, args.train_edge, num_actions, args.weighting_lambda, length_constraint = args.num_frames)
        rewards = get_option_rewards(args.record_rollouts, reward_classes, actions, length_constraint = args.num_frames, raws=raws, dumps=dumps)
        actions, states, resps = generate_trace_training(actions, rewards, states, resps, args.trace_len)
        targets = None
        criteria = supervised_criteria
    elif args.pretrain_target == 1:
        states, resps, num_actions, state_class, proxy_chain, raws, dumps = get_states(args, true_environment, length_constraint = args.num_frames)
        try:
            actions = np.load("actions.npy") # delete this in filesystem when you need new actions
        except FileNotFoundError as e:
            actions = generate_soft_dataset(states, resps, true_environment, reward_classes, args)
            np.save("actions.npy", actions)
        states, resps = [states.copy() for _ in range(len(reward_classes))], [resps.copy() for _ in range(len(reward_classes))]
        criteria = supervised_criteria
        targets = None
    elif args.pretrain_target == 2:
        states, resps, num_actions, state_class, proxy_chain, raws, dumps = get_states(args, true_environment, length_constraint = args.num_frames)
        actions = get_option_actions(args.record_rollouts, proxy_chain[-1].name.split("->")[0], 3, args.weighting_lambda, length_constraint = args.num_frames)
        rewards = get_option_rewards(args.record_rollouts, [BounceReward(0, args), BounceReward(1, args), BounceReward(2, args), BounceReward(3, args)], actions, length_constraint=args.num_frames)
        actions, indexes = generate_distilled_training(rewards)
        num_actions = 4
        print("target training")
        actions, states, resps, targets = generate_target_training(actions, indexes, states, resps, state_class, reward_classes, args.record_rollouts, args.trace_len, num_actions, length_constraint= args.num_frames, raws=raws, dumps=dumps)
        criteria = supervised_criteria
    # elif args.optimizer_form in ["DQN", "SARSA", "Dist"]: # dist might have mode collapse to protect against
    #     pass # TODO: implement
    pretrain(args, true_environment, actions, num_actions, state_class, states, resps, targets, criteria, reward_classes)