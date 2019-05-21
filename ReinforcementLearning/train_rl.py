import os, collections, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import time
import gc, cv2

from Environments.environment_specification import ProxyEnvironment
from Models.models import pytorch_model
from ReinforcementLearning.rollouts import RolloutOptionStorage
from file_management import save_to_pickle

def sample_actions( probs, deterministic): # TODO: why is this here?
    if deterministic is False:
        cat = torch.distributions.categorical.Categorical(probs.squeeze())
        action = cat.sample()
        action = action.unsqueeze(-1).unsqueeze(-1)
    else:
        action = probs.max(1)[1]
    return action


def unwrap_or_none(val):
    if val is not None:
        if type(val) == torch.tensor:
            return pytorch_model.unwrap(val)
        return val
    else:
        return -1.0

def trainRL(args, save_path, true_environment, train_models, learning_algorithm, proxy_environment,
            proxy_chain, reward_classes, state_class, behavior_policy):
    print("#######")
    print("Training Options")
    print("#######")
    # if option_chain is not None: #TODO: implement this
    base_env = proxy_chain[0]
    base_env.set_save(0, args.save_dir, args.save_recycle, args.single_save_dir)
    proxy_environment.initialize(args, proxy_chain, reward_classes, state_class, behavior_policy)
    if args.save_models:
        if args.env.find("Atari") != -1:
            screen = base_env.screen
            base_env.screen = None
        save_to_pickle(os.path.join(save_path, "env.pkl"), proxy_environment)
        if args.env.find("Atari") != -1:
            base_env.screen = screen
    behavior_policy.initialize(args, proxy_environment.action_size)
    print(reward_classes[0], reward_classes[0].parameter_minmax)
    if not args.load_weights:
        train_models.initialize(args, len(reward_classes), state_class, proxy_environment.action_size, parameter_minmax = reward_classes[0].parameter_minmax)
        proxy_environment.set_models(train_models)
    else:
        train_models.initialize(args, len(reward_classes), state_class, proxy_environment.action_size, parameter_minmax = reward_classes[0].parameter_minmax)
        train_models.session(args)
        proxy_environment.duplicate(args)
    train_models.train()
    proxy_environment.set_save(0, args.save_dir, args.save_recycle)
    learning_algorithm.initialize(args, train_models, reward_classes=reward_classes)
    print(proxy_environment.get_names())
    state = pytorch_model.wrap(proxy_environment.getState(), cuda = args.cuda)
    cs, cr = proxy_environment.getHistState()
    hist_state = pytorch_model.wrap(cs, cuda = args.cuda)
    raw_state = base_env.getState()
    resp = proxy_environment.getResp()
    cp_state = proxy_environment.changepoint_state([raw_state])
    # print("initial_state (s, hs, rs, cps)", state, hist_state, raw_state, cp_state)
    # print(cp_state.shape, state.shape, hist_state.shape, state_class.shape)
    print(args.trace_len, args.trace_queue_len)
    rollouts = RolloutOptionStorage(args.num_processes, (state_class.shape,), proxy_environment.action_size, cr.flatten().shape[0],
        state.shape, hist_state.shape, args.buffer_steps, args.changepoint_queue_len, args.trace_len, 
        args.trace_queue_len, args.dilated_stack, args.target_stack, args.dilated_queue_len, train_models.currentOptionParam().shape[1:], len(train_models.models), cp_state[0].shape,
        args.lag_num, args.cuda, return_form = args.return_form)
    option_actions = {option.name: collections.Counter() for option in train_models.models}
    total_duration = 0
    total_elapsed = 0
    true_reward = 0
    ep_reward = 0
    sample_schedule = args.sample_schedule
    start = time.time()
    fcnt = 0
    final_rewards = list()
    average_rewards, average_counts = [], []
    option_counter = collections.Counter()
    option_value = collections.Counter()
    trace_queue = [] # keep the last states until end of trajectory (or until a reset), and dump when a reward is found
    retest = False
    done = False
    for j in range(args.num_iters):
        rollouts.set_parameters(learning_algorithm.current_duration * args.reward_check)            
        # print("set_parameters", state)
        raw_actions = []
        rollouts.cuda()
        last_total_steps, total_steps = 0, 0
        s = time.time()
        for step in range(learning_algorithm.current_duration):

            for m in range(args.reward_check):
                fcnt += 1
                total_steps += 1

                current_state, current_resp = proxy_environment.getHistState()
                estate = proxy_environment.getState()
                values, log_probs, action_probs, Q_vals = train_models.determine_action(current_state.unsqueeze(0), current_resp.unsqueeze(0))
                v, ap, lp, qv = train_models.get_action(values, action_probs, log_probs, Q_vals)

                # a = time.time()
                # print("choose action", a-s)
                action = behavior_policy.take_action(ap, qv)
                cp_state = proxy_environment.changepoint_state([raw_state])
                # print(state, action)
                # print("before_insert", state)
                # cv2.imshow('nextframe',pytorch_model.unwrap(state.view(84,84) / 255.0))
                # if cv2.waitKey(10000) & 0xFF == ord('q'):
                #     pass

                rollouts.insert(retest, state, current_state, pytorch_model.wrap(args.greedy_epsilon, cuda=args.cuda), done, current_resp, action, cp_state[0], train_models.currentOptionParam(), train_models.option_index, None, None, action_probs, Q_vals, values)
                rollouts.insert_dilation(proxy_environment.swap)

                retest = False
                # print("step states (cs, ns, cps, act)", current_state, estate, cp_state, action) 
                # print("step outputs (val, de, ap, qv, v, ap, qv)", values, dist_entropy, action_probs, Q_vals, v, ap, qv)
                trace_queue.append((current_state.clone().detach(), action.clone().detach()))
                state, raw_state, resp, done, action_list = proxy_environment.step(action, model = False)#, render=len(args.record_rollouts) != 0, save_path=args.record_rollouts, itr=fcnt)
                # print(action_list)
                # s = time.time()
                # print("step time", s-a)
                # print("after step", state)
                true_reward += base_env.reward
                ep_reward += base_env.reward
                if args.reward_form == 'raw':
                    for rc in reward_classes:
                        rc.insert_reward(base_env.reward)
                # print(action_list, action)
                # print("step check (al, s)", action_list, state)
                #### logging
                option_actions[train_models.currentName()][int(pytorch_model.unwrap(action.squeeze()))] += 1
                #### logging
                if done:
                    print("Episode Reward: ", ep_reward, " ", fcnt, j)
                    ep_reward = 0
                    if not args.sample_duration > 0 or (args.done_swapping <= j):
                        # print("reached end")
                        # print(step)
                        if args.trace_queue_len > -1:
                            trace_queue = rollouts.insert_trace(trace_queue)

                        trace_queue = []
                        break
                    else: # need to clear out trace queue
                        trace_queue = rollouts.insert_trace(trace_queue)
                        trace_queue = []
                # time.sleep(.1)
            # print(m, args.reward_check)
            # rl = time.time()
            # print("run loop", start - rl)
            rewards = proxy_environment.computeReward(m+1)
            # a = time.time()
            # print("reward time", a-s)
            change, target = proxy_environment.determineChanged(m+1)
            proxy_environment.determine_swaps(m+1, needs_rewards=True) # doesn't need to generate rewards
            # print("reward time", time.time() - start)
            # print("rewards", torch.sum(rewards))

            # reenter to get next value
            current_state, current_resp = proxy_environment.getHistState()
            values, log_probs, action_probs, Q_vals = train_models.determine_action(current_state.unsqueeze(0), current_resp.unsqueeze(0))
            v, ap, lp, qv = train_models.get_action(values, action_probs, log_probs, Q_vals)
            action = behavior_policy.take_action(ap, qv)
            trace_queue.append((current_state.clone().detach(), action.clone().detach()))
            cp_state = proxy_environment.changepoint_state([raw_state])
            rollouts.insert(retest, state, current_state, pytorch_model.wrap(args.greedy_epsilon, cuda=args.cuda), done, current_resp, action, cp_state[0], train_models.currentOptionParam(), train_models.option_index, None, None, action_probs, Q_vals, values) # inserting the last state and unused action
            retest = True # need to re-insert value with true state
            # ########
            rollouts.insert_hindsight_target(change, target)
            rollouts.insert_rewards(args, rewards)
            name = train_models.currentName()
            option_counter[name] += m + 1
            option_value[name] += rewards.sum(dim=1)[train_models.option_index]  

            last_total_steps = total_steps
            completed = learning_algorithm.interUpdateModel(total_steps, rewards, change, done)
            # rw = time.time()
            # print("rewards", rl - rw, start - rw)


            if completed or (done and not args.sample_duration > 0):
                break

        retest = args.buffer_steps > 0 or args.lag_num > 0 # if we roll, don't retest
            # print("steptime", time.time() - start)
        # start = time.time()
        # print(done)
        # print(rollouts.base_rollouts.extracted_state, rollouts.base_rollouts.rewards)
        # print("rew, state", rollouts.rewards[0,-50:], rollouts.extracted_state[-50:])
        # print("inserttime", time.time() - start)
        # print("states and actions (es, cs, a, m)", rollouts.extracted_state, rollouts.current_state, rollouts.actions, rollouts.masks)
        # print("actions and Qvals (qv, vp, ap)", rollouts.Qvals, rollouts.value_preds, rollouts.action_probs)
        # start = time.time()
        total_duration += total_steps
        # if done:
        #     trace_queue = rollouts.insert_trace(trace_queue)
        #     trace_queue = [] # insert first
        # else:
        #     trace_queue = rollouts.insert_trace(trace_queue)
        # print(rollouts.extracted_state)
        # print(rewards)
        # rollouts.compute_returns(args, values) # don't need to compute returns because they are computed upon reward reception
        # print("returns and rewards (rew, ret)", rollouts.rewards, rollouts.returns)
        # print("returns and return queue", rollouts.returns, rollouts.return_queue)
        # print("reward check (cs, rw, rol rw, rt", rollouts.current_state, rewards, rollouts.rewards, rollouts.returns)
        
        # print(name, rollouts.extracted_state, rollouts.rewards, rollouts.actions)
        # n = 0
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj):
        #             n+=1
        #     except:
        #         pass
        # print("learning at", j, n)

        #### logging
        # print(rollouts.base_rollouts.rewards.shape)
        reward_total = rollouts.get_current(names=['rewards'])[0][train_models.option_index].sum(dim=0)
        # print("reward_total", reward_total.shape)
        final_rewards.append(reward_total)
        #### logging
        # start = time.time()
        learning_algorithm.step_counter += 1
        if j >= args.warm_up: # TODO: clean up this to learning algorithm?
            value_loss, action_loss, dist_entropy, output_entropy, entropy_loss, action_log_probs = learning_algorithm.step(args, train_models, rollouts)
            if args.dist_interval != -1 and j % args.dist_interval == 0:
                learning_algorithm.distibutional_sparcity_step(args, train_models, rollouts)
                # print("di", time.time() - start)
            if args.correlate_steps > 0 and j % args.diversity_interval == 0:
                loss= learning_algorithm.correlate_diversity_step(args, train_models, rollouts)
                # print("corr", time.time() - start)
            if args.greedy_epsilon_decay > 0 and j % args.greedy_epsilon_decay == 0 and j != 0:
                behavior_policy.epsilon = max(args.min_greedy_epsilon, behavior_policy.epsilon * 0.5) # TODO: more advanced greedy epsilon methods
                # print("eps", time.time() - start)
            if args.sample_schedule > 0 and j % sample_schedule == 0 and j != 0:
                learning_algorithm.sample_duration = (j // args.sample_schedule + 1) * args.sample_duration
                learning_algorithm.reset_current_duration(learning_algorithm.sample_duration, args.reward_check)
                args.changepoint_queue_len = max(learning_algorithm.max_duration, args.changepoint_queue_len)
                sample_schedule = args.sample_schedule * (j // args.sample_schedule + 1)# sum([args.sample_schedule * (i+1) for i in range(j // args.sample_schedule + 1)])
            if args.retest_schedule > 0 and j % args.retest_schedule == 0 and j != 0:
                learning_algorithm.retest += 1
                learning_algorithm.reset_current_duration(learning_algorithm.sample_duration, args.reward_check)
                args.changepoint_queue_len = max(learning_algorithm.max_duration, args.changepoint_queue_len)
                # print("resample", time.time() - start)
            if j > args.done_swapping:
                learning_algorithm.reset_current_duration(learning_algorithm.sample_duration, args.reward_check)
        else:
            value_loss, action_loss, dist_entropy, output_entropy, entropy_loss, action_log_probs = None, None, None, None, None, None
        parameter = proxy_environment.get_next_parameter()
        if args.reward_swapping:
            parameter = completed    
        learning_algorithm.updateModel(parameter)
        # s = time.time()
        # print("learning step time", s-a)
        # n = 0
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj):
        #             n+=1
        #     except:
        #         pass
        # print("objects at", j, n)

        # print("update", time.time() - start)
        # print("learn time", time.time() - rw)
        if j % args.save_interval == 0 and args.save_models and args.train: # no point in saving if not training
            print("=========SAVING MODELS==========")
            train_models.save(save_path) # TODO: implement save_options


        #### logging
        if j % args.log_interval == 0:
            print("Qvalue and state", pytorch_model.unwrap(Q_vals.squeeze()), pytorch_model.unwrap(current_state.squeeze()))
            print("probs and state", pytorch_model.unwrap(action_probs.squeeze()), pytorch_model.unwrap(current_state.squeeze()))
            for name in train_models.names():
                if option_counter[name] > 0:
                    print(name, option_value[name] / option_counter[name], [option_actions[name][i]/option_counter[name] for i in range(len(option_actions[name]))])
                # if j % (args.log_interval * 20) == 0:
                option_value[name] = 0
                option_counter[name] = 0
                for i in range(len(option_actions[name])):
                    option_actions[name][i] = 0
            end = time.time()
            final_rewards = torch.stack(final_rewards).detach()
            average_rewards.append(final_rewards.sum())
            average_counts.append(total_duration)
            acount = np.sum(average_counts)
            best_reward = true_reward
            true_reward = true_reward / total_steps
            mean_reward = true_reward
            if len(base_env.episode_rewards) > 0:
                true_reward = np.median(base_env.episode_rewards)
                mean_reward = np.mean(base_env.episode_rewards)
                best_reward = np.max(base_env.episode_rewards)
                
            el, vl, al = unwrap_or_none(entropy_loss), unwrap_or_none(value_loss), unwrap_or_none(action_loss)
            total_elapsed += total_duration
            log_stats = "Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {}, value loss {}, policy loss {}, average_reward {}, true_reward median: {}, mean: {}, max: {}".format(j, total_elapsed,
                       int(total_elapsed / (end - start)),
                       final_rewards.mean(),
                       np.median(final_rewards.cpu()),
                       final_rewards.min(),
                       final_rewards.max(), el,
                       vl, al, torch.stack(average_rewards).sum()/acount, true_reward, mean_reward, best_reward  )
            if acount > 300:
                average_counts.pop(0)
                average_rewards.pop(0)
            true_reward = 0.0
            print(log_stats)
            final_rewards = list()
            total_duration = 0
        #### logging
    if args.save_models and args.train: # no point in saving if not training
        print("=========SAVING MODELS==========")
        train_models.save(save_path) # TODO: implement save_options

    proxy_environment.close_files()
