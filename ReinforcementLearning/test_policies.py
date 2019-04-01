import os, collections, time, copy
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

def testRL(args, save_path, true_environment, proxy_chain, proxy_environment, state_class, behavior_policy, reward_classes = None):
    print("#######")
    print("Evaluating Options")
    print("#######")
    # if option_chain is not None: #TODO: implement this
    base_env = proxy_chain[0]
    base_env.set_save(0, args.save_dir, args.save_recycle)
    if reward_classes is not None:
        proxy_environment.reward_fns = reward_classes
    proxy_environment.initialize(args, proxy_chain, proxy_environment.reward_fns, proxy_environment.stateExtractor, behavior_policy)
    print(base_env.save_path)
    behavior_policy.initialize(args, state_class.action_num)
    train_models = proxy_environment.models

    state = pytorch_model.wrap(proxy_environment.getState(), cuda = args.cuda)
    print(state.shape)
    raw_state = base_env.getState()
    hist_state = pytorch_model.wrap(proxy_environment.getHistState(), cuda = args.cuda)
    cp_state = proxy_environment.changepoint_state([raw_state])
    rollouts = RolloutOptionStorage(args.num_processes, (state_class.shape,), state_class.action_num, 
        state.shape, hist_state.shape, args.buffer_steps, args.changepoint_queue_len, args.trace_len, 
        args.trace_queue_len, train_models.num_options, cp_state[0].shape, args.lag_num, args.cuda)
    option_actions = {option.name: collections.Counter() for option in train_models.models}
    total_duration = 0
    start = time.time()
    fcnt = 0
    final_rewards = list()
    option_counter = collections.Counter()
    option_value = collections.Counter()
    raw_states = dict()
    rollouts.set_parameters(args.num_iters * train_models.num_options)
    if args.num_iters > rollouts.changepoint_queue_len:
        rollouts.set_changepoint_queue(args.num_iters)

    for i in range(train_models.num_options):
        train_models.option_index = i
        train_models.currentModel().test = True
        raw_states[train_models.currentName()] = []
        for j in range(args.num_iters):
            fcnt += 1
            raw_actions = []
            rollouts.cuda()
            current_state = proxy_environment.getHistState()
            values, dist_entropy, action_probs, Q_vals = train_models.determine_action(current_state.unsqueeze(0))
            v, ap, qv = train_models.get_action(values, action_probs, Q_vals)
            cp_state = proxy_environment.changepoint_state([raw_state])
            # print(ap, qv)
            action = behavior_policy.take_action(ap, qv)
            rollouts.insert(j, state, current_state, action_probs, action, Q_vals, values, train_models.option_index, cp_state[0], pytorch_model.wrap(args.greedy_epsilon, cuda=args.cuda))
            state, raw_state, done, action_list = proxy_environment.step(action, model = False)#, render=len(args.record_rollouts) != 0, save_path=args.record_rollouts, itr=fcnt)
            raw_states[train_models.currentName()].append(raw_state)
            option_actions[train_models.currentName()][int(pytorch_model.unwrap(action.squeeze()))] += 1
            if done:
                pass
                # print("reached end")
        print(args.num_iters)
        rewards = proxy_environment.computeReward(rollouts, args.num_iters)
        # print(rewards.shape)
        print(rewards.sum())
        rollouts.insert_rewards(rewards, total_duration)
        total_duration += j
        rollouts.compute_returns(args, values)
        rollouts.cpu()
        save_rols = copy.copy(rollouts)
        save_to_pickle(os.path.join(args.save_dir, "rollouts.pkl"), save_rols)

        reward_total = rollouts.rewards.sum(dim=1)[train_models.option_index] / args.num_iters
        print("Rewards for Policy:", reward_total)
