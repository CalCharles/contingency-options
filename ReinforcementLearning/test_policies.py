import os, collections, time, copy, cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from Environments.environment_specification import ProxyEnvironment
from Models.models import pytorch_model
from ReinforcementLearning.rollouts import RolloutOptionStorage
from file_management import save_to_pickle

def testRL(args, save_path, true_environment, proxy_chain, proxy_environment, state_class, behavior_policy, num_actions, reward_classes = None):
    print("#######")
    print("Evaluating Options")
    print("#######")
    # if option_chain is not None: #TODO: implement this
    base_env = proxy_chain[0]
    base_env.set_save(0, args.save_dir, args.save_recycle)
    if reward_classes is not None:
        proxy_environment.reward_fns = reward_classes
    args.changepoint_queue_len = max(args.changepoint_queue_len, args.num_iters)
    proxy_environment.initialize(args, proxy_chain, proxy_environment.reward_fns, state_class, behavior_policy)
    print(base_env.save_path)
    behavior_policy.initialize(args, num_actions)
    train_models = proxy_environment.models
    train_models.initialize(args, len(reward_classes), state_class, num_actions)
    proxy_environment.duplicate(args)
    proxy_environment.set_save(0, args.save_dir, args.save_recycle)
    state = pytorch_model.wrap(proxy_environment.getState(), cuda = args.cuda)
    resp = proxy_environment.getResp()
    print(state.shape)
    raw_state = base_env.getState()
    cs, cr = proxy_environment.getHistState()
    hist_state = pytorch_model.wrap(cs, cuda = args.cuda)
    cp_state = proxy_environment.changepoint_state([raw_state])
    rollouts = RolloutOptionStorage(args.num_processes, (state_class.shape,), proxy_environment.action_size, cr.flatten().shape[0],
        state.shape, hist_state.shape, args.buffer_steps, args.changepoint_queue_len, args.trace_len, 
        args.trace_queue_len, args.dilated_stack, args.target_stack, args.dilated_queue_len, train_models.currentOptionParam().shape[1:], len(train_models.models), cp_state[0].shape,
        args.lag_num, args.cuda)
    option_actions = {option.name: collections.Counter() for option in train_models.models}
    total_duration = 0
    start = time.time()
    fcnt = 0
    final_rewards = list()
    option_counter = collections.Counter()
    option_value = collections.Counter()
    raw_states = dict()
    ep_reward = 0
    rollouts.set_parameters(args.num_iters * train_models.num_options)
    # if args.num_iters > rollouts.changepoint_queue_len:
    #     rollouts.set_changepoint_queue(args.num_iters)
    done = False
    for i in range(train_models.num_options):
        train_models.option_index = i
        train_models.currentModel().test = True
        raw_states[train_models.currentName()] = []

        for j in range(args.num_iters):
            fcnt += 1
            raw_actions = []
            rollouts.cuda()
            current_state, current_resp = proxy_environment.getHistState()
            values, dist_entropy, action_probs, Q_vals = train_models.determine_action(current_state.unsqueeze(0), current_resp.unsqueeze(0))
            v, ap, qv = train_models.get_action(values, action_probs, Q_vals)
            cp_state = proxy_environment.changepoint_state([raw_state])
            ep_reward += base_env.reward
            # print(ap, qv)
            action = behavior_policy.take_action(ap, qv)
            rollouts.insert(False, state, current_state, pytorch_model.wrap(args.greedy_epsilon, cuda=args.cuda), done, current_resp, action, cp_state[0], train_models.currentOptionParam(), train_models.option_index, None, None, action_probs, Q_vals, values)
            state, raw_state, resp, done, action_list = proxy_environment.step(action, model = False)#, render=len(args.record_rollouts) != 0, save_path=args.record_rollouts, itr=fcnt)
            cv2.imshow('frame',raw_state[0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            raw_states[train_models.currentName()].append(raw_state)
            option_actions[train_models.currentName()][int(pytorch_model.unwrap(action.squeeze()))] += 1
            if done:
                print("Episode Reward: ", ep_reward, " ", fcnt)
                ep_reward = 0
                # print("reached end")
            # proxy_environment.determine_swaps(length, needs_rewards=True) # doesn't need to generate rewards

        print(args.num_iters)
        print(action_probs)
        print("Episode Reward: ", ep_reward, " ", fcnt)
        rewards = proxy_environment.computeReward(args.num_iters)
        # print(rewards.shape)
        # print(rewards.sum())
        rollouts.insert_rewards(args, rewards)
        total_duration += j
        save_rols = copy.deepcopy(rollouts)
        if len(args.save_dir) > 0:
            save_to_pickle(os.path.join(args.save_dir, "rollouts.pkl"), save_rols)

        reward_total = rollouts.base_rollouts.rewards.sum(dim=1)[train_models.option_index] / args.num_iters
        print("Rewards for Policy:", reward_total)
