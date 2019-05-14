import torch
import torch.optim as optim
import torch.nn.functional as F
from file_management import load_from_pickle, get_edge, render_dump
import glob, os
import numpy as np
from Models.models import models, pytorch_model
from Environments.multioption import MultiOption
from OptionChain.option_chain import OptionChain
from Environments.state_definition import GetState, compute_minmax, load_states
from ReinforcementLearning.learning_algorithms import PopOptim
import matplotlib.pyplot as plt

def hot_actions(action_data, num_actions):
    for i in range(len(action_data)):
        hot = np.zeros(num_actions)
        hot[int(action_data[i])] = 1
        action_data[i] = hot.tolist()
    return action_data

def smooth_weight(action_data, lmda):
    for i in range(len(action_data)):
        a = np.array(action_data[i])
        top, topi = np.max(a), np.argmax(a)
        diff = top - (1-lmda)
        if diff > 0:
            norm = a.shape[0] - 1
            a += diff / norm
            a[topi] -= diff + diff / norm
        action_data[i] = a.tolist()
    # print(np.array(action_data))
    return action_data

def get_states(args, true_environment, length_constraint=50000, raws = None, dumps = None):
    dataset_path = args.record_rollouts
    changepoint_path = args.changepoint_dir
    option_chain = OptionChain(true_environment, args.changepoint_dir, args.train_edge, args)
    environments = option_chain.initialize(args)
    print(environments)
    proxy_environment = environments.pop(-1)
    head, tail = get_edge(args.train_edge)
    if len(environments) > 1: # there is a difference in the properties of a proxy environment and the true environment
        num_actions = len(environments[-1].reward_fns)
    else:
        num_actions = environments[-1].num_actions
    state_class = GetState(head, state_forms=list(zip(args.state_names, args.state_forms)))
    use_raw = 'raw' in args.state_forms
    state_class.minmax = compute_minmax(state_class, dataset_path)
    states, resps, raws, dumps = load_states(state_class.get_state, dataset_path, length_constraint = length_constraint, use_raw = use_raw, raws=raws, dumps=dumps)
    return states, resps, num_actions, state_class, environments, raws, dumps

def get_option_actions(pth, train_edge, num_actions, weighting_lambda, length_constraint = 50000, use_hot_actions = True):
    head, tail = get_edge(train_edge)
    action_file = open(os.path.join(pth, tail[0] + "_actions.txt"), 'r')
    actions = []
    for act in action_file:
        # print(act, os.path.join(pth, train_edge + "_actions.txt"))
        actions.append(int(act))
        if len(actions) > length_constraint:
            actions.pop(0)
    action_file.close()
    if use_hot_actions:
        actions = hot_actions(actions, num_actions)
        actions = smooth_weight(actions, weighting_lambda)
    return actions

def get_option_rewards(dataset_path, reward_fns, actions, length_constraint=50000, raws= None, dumps = None):
    states, resps, raws, dumps = load_states(reward_fns[0].get_state, dataset_path, length_constraint=length_constraint, raws=raws, dumps=dumps)
    rewards = []
    for reward_fn in reward_fns:
        reward = reward_fn.compute_reward(pytorch_model.wrap(states, cuda=True), actions, None)
        rewards.append(reward.tolist())
    return rewards

def generate_trace_training(actions, rewards, states, resps, num_steps, gamma=.99):
    trace_actions = [[] for i in range(len(rewards))]
    trace_states = [[] for i in range(len(rewards))]
    trace_resps = [[] for i in range(len(rewards))]
    trace_distance = [[] for i in range(len(rewards))]
    trace_returns = [[] for i in range(len(rewards))]
    states = states.tolist()
    recording = [0 for _ in range(len(rewards))]
    for i in reversed(range(len(rewards[0]))):
        a, rs, s, rsp = actions[i], [rewards[j][i] for j in range(len(rewards))], states[i], resps[i]
        for j, r in enumerate(rs):
            if r == 1:
                recording[j] = num_steps
            if recording[j] > 0: # TODO: recording as an unordered collection, could keep as trajectories
                print(a)
                trace_actions[j].append(a)
                trace_states[j].append(s)
                trace_resps[j].append(rsp)
                trace_distance[j].append(num_steps - recording[j])
                # print(np.power(gamma, num_steps - recording[j]))
                trace_returns[j].append(np.power(gamma, num_steps - recording[j]))
                recording[j] -= 1
            elif np.sum(recording) == 0:
                trace_actions[j].append(a)
                trace_states[j].append(s)
                trace_resps[j].append(rsp)
                trace_returns[j].append(np.power(gamma, num_steps - recording[j]))
                trace_distance[j].append(-1)
        # print(a,s)

    trace_actions = [np.array(vec) for vec in trace_actions]
    trace_states = [np.array(vec) for vec in trace_states]
    trace_resps = [np.array(vec) for vec in trace_resps]
    trace_distance = [np.array(vec) for vec in trace_distance]
    trace_returns = [np.array(vec) for vec in trace_returns]
    print([(a.shape, s.shape) for a,s in zip(trace_actions, trace_states)])
    return trace_actions, trace_states, trace_returns, trace_resps

def generate_distilled_training(rewards):
    indexes = []
    all_rewards = np.sum(rewards, axis=0)
    all_indexes = np.where(all_rewards > .5)[0]
    # print(np.array(rewards[0]) > 1.0)
    match_indexes = [np.where(np.array(rewards[i]) >= 1.0)[0] for i in range(len(rewards))]
    # print(all_rewards, rewards)
    actions = []
    ris = [0 for i in range(len(rewards))]
    print(all_indexes.shape, [m.shape for m in match_indexes])
    while True:
        idxes = []
        for i in range(len(rewards)):
            if ris[i] < len(match_indexes[i]):
                idxes.append(match_indexes[i][ris[i]])
            else:
                idxes.append(len(rewards[0]) + 1) # a maximum value
        action = np.argmin(idxes)
        ris[action] += 1
        actions.append(action)
        if np.sum(ris) == len(all_indexes):
            break
    return np.array(actions), all_indexes

def generate_target_training(actions, indexes, states, resps, state_class, reward_fns, dataset_path, num_steps, num_actions, length_constraint=50000, raws=None, dumps = None):
    train_states = []
    train_actions = []
    train_resps = []
    param_targets = []
    indexes = indexes.tolist()
    indexes = [0] + indexes
    indexes.append(len(states))
    rstates, _, raws, dumps = load_states(reward_fns[0].get_state, dataset_path, length_constraint=length_constraint, raws=raws, dumps=dumps)
    change_indexes, param_idx, hindsight_targets = reward_fns[0].state_class.determine_delta_target(rstates)
    i = 0
    ci = change_indexes[i]
    # print(resps.shape, np.array([hindsight_targets[0].shape[0] for _ in range(len(resps))]).shape)
    # resps = np.concatenate((resps, np.array([[hindsight_targets[0].shape[0]] for _ in range(len(resps))])), axis=1)
    for a, idx1, idx2 in zip(actions, indexes[:-1], indexes[1:]):
        while ci < idx1:
            i += 1
            ci = change_indexes[i]
        if ci < idx2: # we hit a block
            # plt.imshow(render_dump(dumps[idx1]))
            # plt.show()
            # plt.imshow(render_dump(dumps[ci]))
            # plt.show()
            train_states += states[idx1:idx2][:num_steps].tolist() # first 10 states after contact
            train_resps += resps[idx1:idx2][:num_steps].tolist()
            train_actions += hot_actions([a], num_actions) * len(states[idx1:idx2][:num_steps])
            param_targets += [hindsight_targets[i].tolist()] * len(states[idx1:idx2][:num_steps])
            print(len(train_states), len(train_resps), len(train_actions), len(param_targets))
    return np.array([train_actions]), np.array([train_states]), np.array([train_resps]), np.array(param_targets)

def generate_soft_dataset(states, resps, true_environment, reward_fns, args):
    pre_load_weights = args.load_weights
    args.load_weights = True
    option_chain = OptionChain(true_environment, args.changepoint_dir, args.train_edge, args)
    print(args.load_weights)
    environments = option_chain.initialize(args)
    proxy_environment = environments.pop(-1)
    proxy_chain = environments
    train_models = proxy_environment.models
    head, tail = get_edge(args.train_edge)
    if len(environments) > 1: # there is a difference in the properties of a proxy environment and the true environment
        num_actions = len(environments[-1].reward_fns)
    else:
        num_actions = environments[-1].num_actions
    state_class = GetState(head, state_forms=list(zip(args.state_names, args.state_forms)))
    proxy_environment.initialize(args, proxy_chain, reward_fns, state_class, behavior_policy=None)

    train_models.initialize(args, len(reward_fns), state_class, num_actions)
    train_models.session(args)
    proxy_environment.duplicate(args) # assumes that we are loading weights
    args.load_weights = pre_load_weights

    soft_actions = [[] for i in range(train_models.num_options)]
    for oidx in range(train_models.num_options):
        train_models.option_index = oidx
        if args.model_form == 'population':
            train_models.currentModel().use_mean = True
        for i in range(len(states) // 30 + 1):
            state = states[i*30:(i+1)*30]
            resp = resps[i*30:(i+1)*30]
            values, dist_entropy, action_probs, Q_vals = train_models.determine_action(pytorch_model.wrap(state, cuda=args.cuda), pytorch_model.wrap(resp, cuda=args.cuda))
            # print (action_probs)
            values, action_probs, Q_vals = train_models.get_action(values, action_probs, Q_vals)
            soft_actions[oidx] += pytorch_model.unwrap(action_probs).tolist()
    print("soft actions", np.sum(np.array(soft_actions[0]), axis=0))
    for i in range(len(soft_actions)):
        soft_actions[i] = smooth_weight(soft_actions[i], args.weighting_lambda)
    return np.array(soft_actions)

def random_actions(args, true_environment):
    # desired = pytorch_model.wrap(np.stack([desired[:] for i in range(len(train_models.models))], axis=1)) # replicating for different policies
    states, num_actions, state_class, proxy_chain = get_states(args, true_environment)
    actions = [np.random.randint(num_actions) for _ in range(args.num_stack, len(states))]
    actions = hot_actions(actions, num_actions)
    states = np.array([states[i-args.num_stack:i].flatten().tolist() for i in range(args.num_stack, len(states))])
    return np.array(actions), states, num_actions, state_class

def range_Qvals(args, true_environment, minmax):
    states, num_actions, state_class, proxy_chain = get_states(args, true_environment)
    minr, maxr = minmax
    Qvals = [[minr + (maxr - minr) * np.random.rand()] * num_actions for _ in range(args.num_stack, len(states))]
    states = np.array([states[i-args.num_stack:i].flatten().tolist() for i in range(args.num_stack, len(states))])
    return np.array(Qvals), states, num_actions, state_class

class CMAES_optimizer():
    def __init__(self, args, train_models):
        self.optimizers = []
        self.solutions = []
        self.weight_sharing = args.weight_sharing
        for i in range(len(train_models.models)):
            if args.load_weights and not args.freeze_initial: # TODO: initialize from non-population model
                xinit = pytorch_model.unwrap(train_models.models[i].mean.get_parameters())
                # TODO: parameter for sigma?
                sigma = 0.6#pytorch_model.unwrap(torch.stack([train_models.models[i].networks[j].get_parameters() for j in range(train_models.models[i].num_population)]).var(dim=1).mean())
                print(xinit, sigma)
            else:
                xinit = (np.random.rand(train_models.currentModel().networks[0].count_parameters())-0.5)*2 # initializes [-1,1]
                sigma = 1.0
            cmaes_params = {"popsize": args.num_population} # might be different than the population in the model...
            cmaes = cma.CMAEvolutionStrategy(xinit, sigma, cmaes_params)
            self.optimizers.append(cmaes)
            self.solutions.append(cmaes.ask())
        for i in range(len(self.models.models)):
            self.assign_solutions(train_models, i)

    def assign_solutions(self, train_models, i):
        for j in range(train_models.models[i].num_population):
            train_models.models[i].networks[j].set_parameters(self.solutions[i][j])

    def step(self, models, values, dist_entropy, action_probs, Q_vals, optimizer, true_values):
        for midx in range(len(models.models)):
            models.option_index = midx
            loss = self.criteria(models, values, dist_entropy, action_probs, Q_vals, optimizer, true_values[midx])
            solutions = self.solutions[option_index]
            # returns = torch.stack([rollout_returns[self.sample_duration * i:self.sample_duration * (i+1)].sum() / self.sample_duration for i in range(args.num_population)])
            cmaes = self.optimizers[models.option_index]
            cmaes.tell(solutions, loss)
            self.solutions[models.option_index] = cmaes.ask()
            self.assign_solutions(models, models.option_index)
            best = cmaes.result[0]
            mean = cmaes.result[5]
            self.models.currentModel().best.set_parameters(best)
            self.models.currentModel().mean.set_parameters(mean)
        return loss.mean()

def supervised_criteria(models, values, dist_entropy, action_probs, Q_vals, optimizer, true_values, targets):
    loss = F.binary_cross_entropy(action_probs.squeeze(), pytorch_model.wrap(true_values, cuda=True).squeeze()) # TODO: cuda support required
    loss += -(action_probs.squeeze() * torch.log(action_probs.squeeze() + 1e-10)).sum(dim=1).mean() * .01
    # print(action_probs[:5], true_values[:5], loss)
    # for optimizer in optimizers:
    #     optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def action_criteria(models, values, dist_entropy, action_probs, Q_vals, optimizer, true_values, targets):
    dist_ent = -(action_probs.squeeze() * torch.log(action_probs.squeeze() + 1e-10)).sum(dim=1).mean()
    batch_mean = action_probs.squeeze().mean(dim=0)
    batch_ent = -((batch_mean + 1e-10) * torch.log(batch_mean + 1e-10)).sum()
    print(batch_ent, dist_ent, batch_mean, action_probs[0][0])
    loss = dist_ent - batch_ent
    # for optimizer in optimizers:
    optimizer.zero_grad()
    loss.backward()
    # for optimizer in optimizers:
    optimizer.step()
    return loss

def Q_criteria(models, values, dist_entropy, action_probs, Q_vals, optimizer, true_values, targets):
    # we should probably include the criteria
    # print(true_values, Q_vals.shape, pytorch_model.wrap(targets, cuda=True).squeeze().long().shape)
    # print(pytorch_model.wrap(targets, cuda=True).squeeze().long())
    # print(Q_vals.gather(1, pytorch_model.wrap(targets, cuda=True).unsqueeze(1).long()))
    loss = (Q_vals.gather(1, pytorch_model.wrap(targets, cuda=True).unsqueeze(1).long()) - pytorch_model.wrap(true_values, cuda=True).squeeze()).pow(2).mean()
    # for optimizer in optimizers:
    optimizer.zero_grad()
    loss.backward()
    # for optimizer in optimizers:
    optimizer.step()
    return loss

def pretrain(args, true_environment, desired, num_actions, state_class, states, resps,  targets, criteria, reward_fns):
    # args = get_args()
    # true_environment = Paddle()
    # true_environment = PaddleNoBlocks()
    dataset_path = args.record_rollouts
    changepoint_path = args.changepoint_dir
    option_chain = OptionChain(true_environment, args.changepoint_dir, args.train_edge, args)
    environments = option_chain.initialize(args)
    proxy_environment = environments.pop(-1)
    proxy_chain = environments
    if args.load_weights:
        train_models = proxy_environment.models
    else:
        train_models = MultiOption(1, models[args.model_form])
    head, tail = get_edge(args.train_edge)
    print(args.state_names, args.state_forms)
    print(state_class.minmax)
    # behavior_policy = EpsilonGreedyProbs()
    save_dir = args.save_graph
    if args.save_graph == "graph":
        save_dir = option_chain.save_dir
    proxy_environment.initialize(args, proxy_chain, reward_fns, state_class, behavior_policy=None)
    fit(args, save_dir, true_environment, train_models, state_class, desired, states, resps, targets, num_actions, criteria, proxy_environment, reward_fns)

def fit(args, save_dir, true_environment, train_models, state_class, desired, states, resps, targets, num_actions, criteria, proxy_environment, reward_classes):
    parameter_minmax = None
    if args.model_form.find("param") != -1:
        parameter_minmax = (np.min(targets, axis=0), np.max(targets, axis=0))
        # state_class.sizes.append(parameter_minmax[0].shape)
    print(parameter_minmax)
    if not args.load_weights:
        state_class.action_num = num_actions
        train_models.initialize(args, len(reward_classes), state_class, num_actions, parameter_minmax=parameter_minmax)
        proxy_environment.set_models(train_models)
    else:
        train_models.initialize(args, len(reward_classes), state_class, num_actions, parameter_minmax=parameter_minmax)
        train_models.session(args)
        proxy_environment.duplicate(args)
    print(state_class.fnames)
    train_models.train()    
    # print(len([desired_actions[:] for i in range(len(train_models.models))]))
    print("train_models", len(train_models.models))
    print("num states", len(states[0]))
    # desired = pytorch_model.wrap(np.stack([desired[:] for i in range(len(train_models.models))], axis=1)) # replicating for different policies
    optimizers = []
    min_batch = 10
    batch_size = args.num_grad_states
    for model in train_models.models:
        if args.model_form == "population":
            optimizers.append(optim.Adam(model.mean.parameters(), args.lr, eps=args.eps, betas=args.betas, weight_decay=args.weight_decay))
        else:
            optimizers.append(optim.Adam(model.parameters(), args.lr, eps=args.eps, betas=args.betas, weight_decay=args.weight_decay))
    if args.model_form == "population": # train the whole population, each individually
        for model in train_models.models:
            model.use_mean = True
    for oidx in range(train_models.num_options):
        print("Training option: ", oidx)
        ostates = states[oidx]
        oresps = resps[oidx]
        odesired = desired[oidx]
        total_loss = 0.0
        for i in range(args.num_iters):
            idxes = np.random.choice(list(range(len(odesired))), (batch_size,), replace=False)
            if args.model_form.find("param") != -1:
                param_vals = targets[idxes]
                train_models.currentModel().option_values = pytorch_model.wrap(param_vals, cuda=args.cuda)
            if args.optimizer_form in ["DQN", "SARSA"]:
                param_vals = targets[oidx][idxes]
                # print(targets[oidx][idxes])
            # start = np.random.randint(len(odesired))
            # idxes = [(idx + start) % len(odesired) for idx in range(batch_size)]
            # print(pytorch_model.wrap(ostates[idxes], cuda=args.cuda), pytorch_model.wrap(oresps[idxes], cuda=args.cuda))
            values, dist_entropy, action_probs, Q_vals = train_models.determine_action(pytorch_model.wrap(ostates[idxes], cuda=args.cuda), pytorch_model.wrap(oresps[idxes], cuda=args.cuda))
            values, action_probs, Q_vals = train_models.get_action(values, action_probs, Q_vals)
            # print(action_probs.transpose(1,0).shape, desired[idxes].shape)
            # print(action_probs.squeeze().shape, desired.shape)
            # print(states[idxes])
            # print(action_probs,(action_probs.squeeze() * torch.log(action_probs.squeeze())).sum(dim=1).mean())
            # print(train_models.currentModel().mean.action_probs.weight)
            loss = criteria(train_models, values, dist_entropy, action_probs, Q_vals, optimizers[oidx], odesired[idxes], param_vals)
            # print(train_models.currentModel().mean.action_probs.weight)
            total_loss += loss.detach()
            if i % args.log_interval == 0:
                print(action_probs[:5])
                print(ostates[idxes][:5])
                print(odesired[idxes][:5])
                print("iter ", i, " at loss: ", total_loss.detach().cpu() / args.log_interval)
                total_loss = 0
            if args.sample_schedule > 0 and i % (args.sample_schedule) == 0:
                batch_size = max(batch_size //   2, min_batch)
            if i % args.save_interval == 0 and args.save_models: # no point in saving if not training
                print("=========SAVING MODELS==========")
                train_models.save(save_dir) # TODO: implement save_options???? DONE??
        train_models.option_index += 1
    if args.model_form == "population": # train the whole population, each individually
        for model in train_models.models:
            model.use_mean = True

    if args.save_models:
        train_models.save(save_dir) # TODO: implement save_options