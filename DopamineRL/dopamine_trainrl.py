import os, collections, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from torch.autograd import Variable
import torch.optim as optim

from Environments.environment_specification import ProxyEnvironment
from Models.models import pytorch_model
from file_management import save_to_pickle

def sample_actions( probs, deterministic):
    if deterministic is False:
        cat = torch.distributions.categorical.Categorical(probs.squeeze())
        action = cat.sample()
        action = action.unsqueeze(-1).unsqueeze(-1)
    else:
        action = probs.max(1)[1]
    return action


def unwrap_or_none(val):
    if val is not None:
        return pytorch_model.unwrap(val)
    else:
        return -1.0

def train_dopamine(args, save_path, true_environment, train_models, proxy_environment,
            proxy_chain, reward_classes, state_class, num_actions, behavior_policy):
    print("#######")
    print("Training Options")
    print("#######")
    # if option_chain is not None: #TODO: implement this
    base_env = proxy_chain[0]
    base_env.set_save(0, args.save_dir, args.save_recycle)
    snum = args.num_stack 
    args.num_stack = 1
    proxy_environment.initialize(args, proxy_chain, reward_classes, state_class, behavior_policy)
    args.num_stack = snum
    if args.save_models:
        save_to_pickle(os.path.join(save_path, "env.pkl"), proxy_environment)
    behavior_policy.initialize(args, num_actions)
    train_models.initialize(args, len(reward_classes), state_class, proxy_environment.action_size)
    proxy_environment.set_models(train_models)
    proxy_environment.set_save(0, args.save_dir, args.save_recycle)
    state = pytorch_model.wrap(proxy_environment.getState(), cuda = args.cuda)
    cs, cr = proxy_environment.getHistState()
    hist_state = pytorch_model.wrap(cs, cuda = args.cuda)
    raw_state = base_env.getState()
    cp_state = proxy_environment.changepoint_state([raw_state])
    # print("initial_state (s, hs, rs, cps)", state, hist_state, raw_state, cp_state)
    # print(cp_state.shape, state.shape, hist_state.shape, state_class.shape)
    # rollouts = RolloutOptionStorage(args.num_processes, (state_class.shape,), proxy_environment.action_size, cr.flatten().shape[0],
    #     state.shape, hist_state.shape, args.buffer_steps, args.changepoint_queue_len, args.trace_len, 
    #     args.trace_queue_len, args.dilated_stack, args.target_stack, args.dilated_queue_len, train_models.currentOptionParam().shape[1:], len(train_models.models), cp_state[0].shape,
    #     args.lag_num, args.cuda)
    option_actions = {option.name: collections.Counter() for option in train_models.models}
    total_duration = 0
    total_elapsed = 0
    true_reward = 0
    ep_reward = 0
    start = time.time()
    fcnt = 0
    final_rewards = list()
    option_counter = collections.Counter()
    option_value = collections.Counter()
    print(hist_state)
    val= None
    train_models.currentModel().begin_episode(pytorch_model.unwrap(hist_state))
    for j in range(args.num_iters):
        raw_actions = []
        last_total_steps, total_steps = 0, 0
        for step in range(args.num_steps):
            # start = time.time()
            fcnt += 1
            total_steps += 1
            current_state, current_resp = proxy_environment.getHistState()
            estate = proxy_environment.getState()
            if args.true_environment:
                reward = pytorch_model.wrap([[base_env.reward]])
            else:
                reward = proxy_environment.computeReward(1)
            true_reward += base_env.reward
            ep_reward += base_env.reward
            # print(current_state, reward[train_models.option_index])
            action = train_models.currentModel().forward(current_state, pytorch_model.unwrap(reward[train_models.option_index]))
            # print("ap", action)
            action = pytorch_model.wrap([action])
            cp_state = proxy_environment.changepoint_state([raw_state])
            # print(state, action)
            # print("step states (cs, s, cps, act)", current_state, estate, cp_state, action) 
            # print("step outputs (val, de, ap, qv, v, ap, qv)", values, dist_entropy, action_probs, Q_vals, v, ap, qv)

            state, raw_state, resp, done, action_list = proxy_environment.step(action, model = False)#, render=len(args.record_rollouts) != 0, save_path=args.record_rollouts, itr=fcnt)
            # print("step check (al, s)", action_list, state)
            # learning_algorithm.interUpdateModel(step)
            #### logging
            option_actions[train_models.currentName()][int(pytorch_model.unwrap(action.squeeze()))] += 1
            #### logging
            # print(train_models.currentModel().dope_rainbow)

            if done:
                # print("reached end")
                print("Episode Reward: ", ep_reward)
                ep_reward = 0
                train_models.currentModel().end_episode(pytorch_model.unwrap(reward[train_models.option_index]))
                state, resp = proxy_environment.getHistState()
                train_models.currentModel().begin_episode(pytorch_model.unwrap(state))
                # print(step)
                break
        # var = [v for v in tf.trainable_variables() if v.name == "Online/fully_connected/weights:0"][0]
        # nval = train_models.currentModel().sess.run(var)
        # if val is not None: 
        #     print(var, np.sum(abs(nval - val)), train_models.currentModel().dope_rainbow.eval_mode)
        # val = nval
        current_state = proxy_environment.getHistState()
        # print(state, action)
        # print("step states (cs, s, cps, act)", current_state, estate, cp_state, action) 
        # print("step outputs (val, de, ap, qv, v, ap, qv)", values, dist_entropy, action_probs, Q_vals, v, ap, qv)

        cp_state = proxy_environment.changepoint_state([raw_state])
        # print("states and actions (es, cs, a, m)", rollouts.extracted_state, rollouts.current_state, rollouts.actions, rollouts.masks)
        # print("actions and Qvals (qv, vp, ap)", rollouts.Qvals, rollouts.value_preds, rollouts.action_probs)

        total_duration += step + 1
        # print("rewards", rewards)
        # rollouts.insert_rewards(rewards)
        # print(rollouts.extracted_state)
        # print(rewards)
        # rollouts.compute_returns(args, values)
        # print("returns and rewards (rew, ret)", rollouts.rewards, rollouts.returns)
        # print("returns and return queue", rollouts.returns, rollouts.return_queue)
        # print("reward check (cs, rw, rol rw, rt", rollouts.current_state, rewards, rollouts.rewards, rollouts.returns)
        name = train_models.currentName()
        # print(name, rollouts.extracted_state, rollouts.rewards, rollouts.actions)

        #### logging
        option_counter[name] += step + 1
        option_value[name] += true_reward 
        #### logging
        if j % args.save_interval == 0 and args.save_models and args.train: # no point in saving if not training
            print("=========SAVING MODELS==========")
            train_models.save(save_path) # TODO: implement save_options


        #### logging
        if j % args.log_interval == 0:
            # print("Qvalue and state", pytorch_model.unwrap(Q_vals.squeeze()), pytorch_model.unwrap(current_state.squeeze()))
            # print("probs and state", pytorch_model.unwrap(action_probs.squeeze()), pytorch_model.unwrap(current_state.squeeze()))
            for name in train_models.names():
                if option_counter[name] > 0:
                    print(name, option_value[name] / option_counter[name], [option_actions[name][i]/option_counter[name] for i in range(len(option_actions[name]))])
                if j % (args.log_interval * 20) == 0:
                    option_value[name] = 0
                    option_counter[name] = 0
                    for i in range(len(option_actions[name])):
                        option_actions[name][i] = 0
            end = time.time()
            total_elapsed += total_duration
            log_stats = "Updates {}, num timesteps {}, FPS {}, reward {}".format(j, total_elapsed,
                       int(total_elapsed / (end - start)),
                       true_reward/ (args.num_steps * args.log_interval))
            print(log_stats)
            true_reward = 0.0
            total_duration = 0
        #### logging
