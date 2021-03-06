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
    args.changepoint_queue_len = max(args.changepoint_queue_len, args.num_iters * args.num_update_model)
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
    rollouts.set_parameters(args.num_iters * args.num_update_model)
    # if args.num_iters > rollouts.changepoint_queue_len:
    #     rollouts.set_changepoint_queue(args.num_iters)
    done = False
    ctr = 0
    raw_indexes = dict()
    for i in range(args.num_iters):
        train_models.option_index = np.random.randint(train_models.num_options)
        train_models.currentModel().test = True
        if train_models.currentName() not in raw_states:
            raw_states[train_models.currentName()] = []
            raw_indexes[train_models.currentName()] = []

        for j in range(args.num_update_model):
            raw_indexes[train_models.currentName()].append(ctr)
            ctr += 1
            fcnt += 1
            raw_actions = []
            rollouts.cuda()
            current_state, current_resp = proxy_environment.getHistState()
            values, log_probs , action_probs, Q_vals = train_models.determine_action(current_state.unsqueeze(0), current_resp.unsqueeze(0))
            v, ap, lp, qv = train_models.get_action(values, action_probs, log_probs, Q_vals)
            cp_state = proxy_environment.changepoint_state([raw_state])
            ep_reward += base_env.reward
            # print(ap, qv)
            action = behavior_policy.take_action(ap, qv)
            # print(train_models.currentName(), action, qv.squeeze())
            rollouts.insert(False, state, current_state, pytorch_model.wrap(args.greedy_epsilon, cuda=args.cuda), done, current_resp, action, cp_state[0], train_models.currentOptionParam(), train_models.option_index, None, None, action_probs, Q_vals, values)
            state, raw_state, resp, done, action_list = proxy_environment.step(action, model = False)#, render=len(args.record_rollouts) != 0, save_path=args.record_rollouts, itr=fcnt)
            print(train_models.currentName(), j, action)
            cv2.imshow('frame',raw_state[0])
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
            raw_states[train_models.currentName()].append(raw_state)
            option_actions[train_models.currentName()][int(pytorch_model.unwrap(action.squeeze()))] += 1
            if done:
                print("Episode Reward: ", ep_reward, " ", fcnt)
                ep_reward = 0
                # print("reached end")
            # proxy_environment.determine_swacurrent_durationps(length, needs_rewards=True) # doesn't need to generate rewards
        # rewards = proxy_environment.computeReward(args.num_update_model)
        # print(rewards)
    if len(base_env.episode_rewards) > 0:
        true_reward = np.median(base_env.episode_rewards)
        mean_reward = np.mean(base_env.episode_rewards)
        best_reward = np.max(base_env.episode_rewards)
        print("true reward median: %f, mean: %f, max: %f"%(true_reward, mean_reward, best_reward))

    print(args.num_iters)
    print(action_probs)
    print("Episode Reward: ", ep_reward, " ", fcnt)
    print(proxy_environment.reward_fns)
    rewards = proxy_environment.computeReward(args.num_iters * args.num_update_model)
    # print(rewards.shape)
    # print(rewards.sum())
    rollouts.insert_rewards(args, rewards)
    total_duration += j
    save_rols = copy.deepcopy(rollouts)
    if len(args.save_dir) > 0:
        save_to_pickle(os.path.join(args.save_dir, "rollouts.pkl"), save_rols)

    for i in range(train_models.num_options):
        print(rollouts.base_rollouts.rewards.shape, raw_indexes)
        reward_total = rollouts.base_rollouts.rewards.sum(dim=1)[i] / (args.num_iters * args.num_update_model)
        # print(rollouts.base_rollouts.rewards, raw_indexes, rollouts.base_rollouts.rewards.shape)
        reward_adjusted = rollouts.base_rollouts.rewards[i, np.array(raw_indexes[train_models.models[i].name]) + args.num_stack].sum(dim=0) / len(raw_indexes[train_models.models[i].name])
        print("Num policy steps:", len(raw_indexes[train_models.models[i].name]))
        print("Rewards during Policy:", reward_adjusted)
        print("Rewards for Policy:", reward_total)

