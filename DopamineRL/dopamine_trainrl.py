import os, collections, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from torch.autograd import Variable
import torch.optim as optim

from Environments.environment_specification import ProxyEnvironment
from ReinforcementLearning.models import pytorch_model
from ReinforcementLearning.rollouts import RolloutOptionStorage
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
            proxy_chain, reward_classes, state_class, behavior_policy):
    print("#######")
    print("Training Options")
    print("#######")
    # if option_chain is not None: #TODO: implement this
    base_env = proxy_chain[0]
    base_env.set_save(0, args.save_dir, args.save_recycle)
    proxy_environment.initialize(args, proxy_chain, reward_classes, state_class, behavior_policy)
    if args.save_models:
        save_to_pickle(os.path.join(save_path, "env.pkl"), proxy_environment)
    behavior_policy.initialize(args, state_class.action_num)
    train_models.initialize(args, len(reward_classes), state_class)
    proxy_environment.set_models(train_models)
    state = pytorch_model.wrap(proxy_environment.getState(), cuda = args.cuda)
    hist_state = pytorch_model.wrap(proxy_environment.getHistState(), cuda = args.cuda)
    raw_state = base_env.getState()
    cp_state = proxy_environment.changepoint_state([raw_state])
    # print("initial_state (s, hs, rs, cps)", state, hist_state, raw_state, cp_state)
    # print(cp_state.shape, state.shape, hist_state.shape, state_class.shape)
    rollouts = RolloutOptionStorage(args.num_processes, (state_class.shape,), state_class.action_num, state.shape, hist_state.shape, -1, args.changepoint_queue_len, len(train_models.models), cp_state[0].shape)
    option_actions = {option.name: collections.Counter() for option in train_models.models}
    total_duration = 0
    total_elapsed = 0
    start = time.time()
    fcnt = 0
    final_rewards = list()
    option_counter = collections.Counter()
    option_value = collections.Counter()
    print(hist_state)
    val= None
    train_models.currentModel().begin_episode(pytorch_model.unwrap(hist_state))
    for j in range(args.num_iters):
        rollouts.set_parameters(args.num_steps)            
        raw_actions = []
        rollouts.cuda()
        for step in range(args.num_steps):
            fcnt += 1
            current_state, current_resp = proxy_environment.getHistState()
            estate = proxy_environment.getState()
            reward = proxy_environment.computeReward(rollouts, 1)
            # print(current_state, reward[train_models.option_index])
            action = train_models.currentModel().forward(current_state, current_resp, pytorch_model.unwrap(reward[train_models.option_index]))
            # print("ap", action)
            action = pytorch_model.wrap([action])
            cp_state = proxy_environment.changepoint_state([raw_state])
            # print(state, action)
            rollouts.insert_no_out(step, state, current_state, action, train_models.option_index, cp_state[0], pytorch_model.wrap(args.greedy_epsilon, cuda=args.cuda))
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
                rollouts.cut_current(step+1)
                reward = proxy_environment.computeReward(rollouts, 1)
                train_models.currentModel().end_episode(pytorch_model.unwrap(reward[train_models.option_index]))
                train_models.currentModel().begin_episode(pytorch_model.unwrap(proxy_environment.getHistState()))
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
        rollouts.insert_no_out(step + 1, state, current_state, action, train_models.option_index, cp_state, pytorch_model.wrap(args.greedy_epsilon, cuda=args.cuda)) # inserting the last state and unused action
        # print("states and actions (es, cs, a, m)", rollouts.extracted_state, rollouts.current_state, rollouts.actions, rollouts.masks)
        # print("actions and Qvals (qv, vp, ap)", rollouts.Qvals, rollouts.value_preds, rollouts.action_probs)

        total_duration += step + 1
        rewards = proxy_environment.computeReward(rollouts, step+1)
        # print("rewards", rewards)
        rollouts.insert_rewards(rewards)
        # print(rollouts.extracted_state)
        # print(rewards)
        # rollouts.compute_returns(args, values)
        # print("returns and rewards (rew, ret)", rollouts.rewards, rollouts.returns)
        # print("returns and return queue", rollouts.returns, rollouts.return_queue)
        # print("reward check (cs, rw, rol rw, rt", rollouts.current_state, rewards, rollouts.rewards, rollouts.returns)
        name = train_models.currentName()
        # print(name, rollouts.extracted_state, rollouts.rewards, rollouts.actions)

        #### logging
        reward_total = rollouts.rewards.sum(dim=1)[train_models.option_index]
        final_rewards.append(reward_total)
        option_counter[name] += step + 1
        option_value[name] += reward_total.data  
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
            final_rewards = torch.stack(final_rewards)
            el, vl, al = unwrap_or_none(pytorch_model.wrap([0])), unwrap_or_none(pytorch_model.wrap([0])), unwrap_or_none(pytorch_model.wrap([0]))
            total_elapsed += total_duration
            log_stats = "Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {}, value loss {}, policy loss {}".format(j, total_elapsed,
                       int(total_elapsed / (end - start)),
                       final_rewards.mean(),
                       np.median(final_rewards.cpu()),
                       final_rewards.min(),
                       final_rewards.max(), el,
                       vl, al)
            print(log_stats)
            final_rewards = list()
            total_duration = 0
        #### logging
