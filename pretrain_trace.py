from SelfBreakout.paddle import Paddle
from SelfBreakout.noblock import PaddleNoBlocks
from arguments import get_args
import numpy as np
from Models.pretrain_models import random_actions, action_criteria, pretrain, range_Qvals,\
                                 Q_criteria, get_states, get_option_actions, get_option_rewards, \
                                 generate_trace_training, generate_soft_dataset, supervised_criteria, \
                                 get_option_rewards, generate_distilled_training, generate_target_training
from RewardFunctions.dummy_rewards import BounceReward, Xreward, BlockReward
import torch 



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
    # ../datasets/caleb_data/bounce/bounce0/ --save-models --save-graph data/bounceOoO/
    # python pretrain_trace.py --model-form basic --record-rollouts "../datasets/caleb_data/bounce/bounce0/" --train-edge "Paddle->Ball" --num-stack 1 --train --num-iters 5000000 --state-forms prox vel --state-names Paddle Ball --base-node Paddle --changepoint-dir ./data/paddlegraph/ --factor 16 --num-layers 2 --lr 1e-8 --init-form xnorm --optimizer-form PPO --reward-form bounce --pretrain-target 1 --weighting-lambda 0.001 --log-interval 2000 --num-frames 1000000 --num-grad-states 30 --save-models --save-graph data/prenet0/ --save-interval 1 > outpre.txt
    # python pretrain_trace.py --model-form multihead --record-rollouts "../datasets/caleb_data/vector/" --train-edge "Paddle->Ball" --num-stack 1 --train --num-iters 1000000 --state-forms prox vel bounds --state-names Paddle Ball Ball --base-node Paddle --changepoint-dir ./data/paddlegraph/ --factor 8 --num-layers 2 --lr 1e-8 --init-form xnorm --optimizer-form PPO --reward-form bounce --pretrain-target 0 --weighting-lambda 0.001 --log-interval 200 --num-frames 50000 --num-grad-states 10 --attention-form vector --key-dim 64 --post-transform-form basic --num-heads 10 --save-models --save-interval 2000 --save-graph ./data/multiheadattnpretrain > pretrainmultivec.txt
    # python pretrain_trace.py --model-form paramcont --record-rollouts "../datasets/caleb_data/transformer_bounce/" --train-edge "Ball->Block" --num-stack 1 --train --num-iters 1000000 --state-forms bin vel bounds --state-names Block Ball Ball --base-node Paddle --changepoint-dir ./data/paddlegraph/ --factor 8 --num-layers 2 --lr 1e-8 --init-form xnorm --optimizer-form PPO --reward-form block --pretrain-target 2 --weighting-lambda 0.001 --log-interval 200 --num-frames 300000 --num-grad-states 10 --key-dim 160 --post-transform-form basic --parameterized-form vector  --parameterized-option 2 --save-models --save-interval 2000 --save-graph ./data/multiheadattnpretrain > pretrainmultivec.txt
    args = get_args()
    torch.cuda.set_device(args.gpu)

    if args.reward_form == 'x':
        reward_classes = [Xreward(args)]
    elif args.reward_form == 'bounce':
        reward_classes = [BounceReward(-1, args)]
    elif args.reward_form == 'dir':
        reward_classes = [BounceReward(0, args), BounceReward(1, args), BounceReward(2, args), BounceReward(3, args)]
    elif args.reward_form == 'block':
        reward_classes = [BlockReward(args)]
    true_environment = Paddle()
    if args.pretrain_target == 0: # pretrain to follow the trace
        states, resps, num_actions, state_class, proxy_chain, raws, dumps = get_states(args, true_environment, length_constraint = args.num_frames)
        actions = get_option_actions(args.record_rollouts, args.train_edge, num_actions, args.weighting_lambda, length_constraint = args.num_frames, use_hot_actions = args.optimizer_form not in ["DQN", "SARSA", "Dist"])
        rewards = get_option_rewards(args.record_rollouts, reward_classes, actions, length_constraint = args.num_frames, raws=raws, dumps=dumps)
        actions, states, q_returns, resps = generate_trace_training(actions, rewards, states, resps, args.trace_len)
        # print(actions.shape, states.shape, resps.shape)
        targets = None
        if args.optimizer_form in ["DQN", "SARSA", "Dist"]:
            desired = q_returns
            targets = actions
            criteria = Q_criteria
        else:
            desired = actions
            criteria = supervised_criteria
    elif args.pretrain_target == 1: # pretrain soft actions
        states, resps, num_actions, state_class, proxy_chain, raws, dumps = get_states(args, true_environment, length_constraint = args.num_frames)
        try:
            actions = np.load("actions.npy") # delete this in filesystem when you need new actions
        except FileNotFoundError as e:
            actions = generate_soft_dataset(states, resps, true_environment, reward_classes, args)
            np.save("actions.npy", actions)
        states, resps = [states.copy() for _ in range(len(reward_classes))], [resps.copy() for _ in range(len(reward_classes))]
        criteria = supervised_criteria
        targets = None
        desired = actions

    elif args.pretrain_target == 2:
        states, resps, num_actions, state_class, proxy_chain, raws, dumps = get_states(args, true_environment, length_constraint = args.num_frames)
        actions = get_option_actions(args.record_rollouts, proxy_chain[-1].name, 3, args.weighting_lambda, length_constraint = args.num_frames)
        rewards = get_option_rewards(args.record_rollouts, [BounceReward(0, args), BounceReward(1, args), BounceReward(2, args), BounceReward(3, args)], actions, length_constraint=args.num_frames)
        actions, indexes = generate_distilled_training(rewards)
        num_actions = 4
        print("target training")
        print(state_class.fnames)
        print(resps)
        actions, states, resps, targets = generate_target_training(actions, indexes, states, resps, state_class, reward_classes, args.record_rollouts, args.trace_len, num_actions, length_constraint= args.num_frames, raws=raws, dumps=dumps)
        criteria = supervised_criteria
        desired = actions

    # elif args.optimizer_form in ["DQN", "SARSA", "Dist"]: # dist might have mode collapse to protect against
    #     pass # TODO: implement
    pretrain(args, true_environment, desired, num_actions, state_class, states, resps, targets, criteria, reward_classes)