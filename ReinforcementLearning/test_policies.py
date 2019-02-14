import os, collections, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from Environments.environment_specification import ProxyEnvironment
from ReinforcementLearning.models import pytorch_model
from ReinforcementLearning.rollouts import RolloutOptionStorage
from file_management import save_to_pickle

def testRL(args, save_path, true_environment, proxy_chain, state_class, behavior_policy):
    print("#######")
    print("Evaluating Options")
    print("#######")
    # if option_chain is not None: #TODO: implement this
    proxy_environment = proxy_chain[-1] #ProxyEnvironment(args, proxy_chain, reward_classes, state_class)
    base_environment = proxy_environment.proxy_chain[0]
    print(base_environment.save_path)
    base_environment.set_save(0, args.save_dir)
    behavior_policy.initialize(args, state_class.action_num)
    train_models = proxy_environment.models
    state = pytorch_model.wrap(proxy_environment.getState(), cuda = args.cuda)
    print(state.shape)
    raw_state = base_environment.getState()
    hist_state = pytorch_model.wrap(proxy_environment.getHistState(), cuda = args.cuda)
    cp_state = proxy_environment.changepoint_state([raw_state])
    rollouts = RolloutOptionStorage(args.num_processes, (state_class.shape,), state_class.action_num, state.shape, hist_state.shape, args.buffer_steps, args.changepoint_queue_len, len(train_models.models), cp_state.shape)
    option_actions = {option.name: collections.Counter() for option in train_models.models}
    total_duration = 0
    total_elapsed = 0
    start = time.time()
    fcnt = 0
    final_rewards = list()
    option_counter = collections.Counter()
    option_value = collections.Counter()
    raw_states = dict()
    for i in range(train_models.num_options):
        train_models.option_index = i
        raw_states[train_models.currentName()] = []
        rollouts.set_parameters(args.num_iters)
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
            rollouts.insert(j, state, current_state, action_probs, action, Q_vals, values, train_models.option_index, cp_state[0])
            state, raw_state, done, action_list = proxy_environment.step(action, model = False)#, render=len(args.record_rollouts) != 0, save_path=args.record_rollouts, itr=fcnt)
            raw_states[train_models.currentName()].append(raw_state)
            option_actions[train_models.currentName()][int(pytorch_model.unwrap(action.squeeze()))] += 1
            if done:
                pass
                # print("reached end")

        rewards = proxy_environment.computeReward(rollouts, args.num_iters)
        # print(rewards)
        rollouts.insert_rewards(rewards)
        rollouts.compute_returns(args, values)
        rollouts.cpu()
        save_rols = copy.copy(rollouts)
        save_to_pickle(os.path.join(args.save_dir, "rollouts.pkl"), save_rols)

        reward_total = rollouts.rewards.sum(dim=1)[train_models.option_index]
        print("Rewards for Policy 1:", reward_total)
