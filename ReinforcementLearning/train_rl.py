import os
from Environments.environment_specification import ProxyEnvironment

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

def trainRL(args, true_environment, train_models, learning_algorithm, 
            proxy_chain, reward_classes, state_class, behavior_policy):
    print("#######")
    print("Training Options")
    print("#######")
    # if option_chain is not None: #TODO: implement this
    save_path = os.path.join(args.save_dir, args.unique_id)
    proxy_environment = ProxyEnvironment(args, proxy_chain, reward_classes, state_class)
    behavior_policy.initialize(args, state_class.action_size)
    train_models.initialize(args, len(reward_classes), state_class)
    proxy_environment.set_models(train_models)
    learning_algorithm.initialize(args, train_models)
    state = pytorch_model.wrap(proxy_environment.getState(), cuda = args.cuda)
    print(state.shape)
    hist_state = pytorch_model.wrap(proxy_environment.getHistState(), cuda = args.cuda)
    rollouts = RolloutOptionStorage(args.num_processes, state_class.state_size, state_class.action_size, state.shape, hist_state.shape, args.buffer_steps, args.changepoint_queue_len, len(train_models.models))
    option_actions = {option.name: collections.Counter() for option in train_models.models}
    total_duration = 0
    total_elapsed = 0
    start = time.time()
    fcnt = 0
    final_rewards = list()
    option_counter = collections.Counter()
    option_value = collections.Counter()
    for j in range(args.num_iters):
        rollouts.set_parameters(learning_algorithm.current_duration)
        raw_actions = []
        rollouts.cuda()
        for step in range(learning_algorithm.current_duration):
            fcnt += 1
            current_state = proxy_environment.getHistState()
            values, dist_entropy, action_probs, Q_vals = train_models.determine_action(current_state)
            action = behavior_policy.take_action(action_probs, Q_vals)
            rollouts.insert(step, state, current_state, action_probs, action, Q_vals, values, train_models.option_index)
            state, raw_state, done = proxy_environment.step(action, model = False)#, render=len(args.record_rollouts) != 0, save_path=args.record_rollouts, itr=fcnt)
            learning_algorithm.interUpdateModel(step)

            #### logging
            option_actions[train_models.currentName()][int(pytorch_model.unwrap(action.squeeze()))] += 1
            #### logging

            if done:
                print("reached end")
                break
        current_state = proxy_environment.getHistState()
        values, dist_entropy, action_probs, Q_vals = train_models.determine_action(current_state)
        action = behavior_policy.take_action(action_probs, Q_vals)
        rollouts.insert(step + 1, state, current_state, action_probs, action, Q_vals, values, train_models.option_index) # inserting the last state and unused action
        total_duration += step + 1
        rewards = proxy_environment.computeReward(rollouts)
        rollouts.insert_rewards(rewards)
        rollouts.compute_returns(args, values)
        name = train_models.currentName()

        #### logging
        reward_total = rollouts.rewards.sum(dim=1)[train_models.option_index]
        final_rewards.append(reward_total)
        option_counter[name] += step + 1
        option_value[name] += reward_total.data  
        #### logging

        value_loss, action_loss, dist_entropy, output_entropy, entropy_loss, action_log_probs = learning_algorithm.step(args, train_models, rollouts) 
        learning_algorithm.updateModel()
        if j % args.save_interval == 0 and args.save_dir != "" and args.train: # no point in saving if not training
            train_models.save(args) # TODO: implement save_options

        #### logging
        if j % args.log_interval == 0:
            for name in train_models.names():
                if option_counter[name] > 0:
                    print(name, option_value[name] / option_counter[name], [option_actions[name][i]/option_counter[name] for i in range(len(option_actions[name]))])
                if j % (args.log_interval * 20) == 0:
                    option_value[name] = 0
                    option_counter[name] = 0
                    for i in range(len(option_actions[name])):
                        option_actions[name][i] = 0
            end = time.time()
            final_rewards = np.array(final_rewards)
            el, vl, al = unwrap_or_none(entropy_loss), unwrap_or_none(value_loss), unwrap_or_none(action_loss)
            total_elapsed += total_duration
            log_stats = "Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {}, value loss {}, policy loss {}".format(j, total_duration,
                       int(total_elapsed / (end - start)),
                       final_rewards.mean(),
                       np.median(final_rewards),
                       final_rewards.min(),
                       final_rewards.max(), el,
                       vl, al)
            print(log_stats)
            final_rewards = list()
            total_duration = 0
        #### logging
