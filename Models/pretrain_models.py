import torch
import torch.optim as optim
import torch.nn.functional as F
from file_management import load_from_pickle, get_edge
import glob, os
import numpy as np
from Models.models import models, pytorch_model
from Environments.multioption import MultiOption
from OptionChain.option_chain import OptionChain
from Environments.state_definition import GetState, compute_minmax, load_states

def hot_actions(action_data, num_actions):
    for i in range(len(action_data)):
        hot = np.zeros(num_actions)
        hot[int(action_data[i])] = 1
        action_data[i] = hot.tolist()
    return action_data

def get_states(args, true_environment):
    dataset_path = args.record_rollouts
    changepoint_path = args.changepoint_dir
    option_chain = OptionChain(true_environment, args.changepoint_dir, args.train_edge, args)
    environments = option_chain.initialize(args)
    proxy_environment = environments.pop(-1)
    head, tail = get_edge(args.train_edge)
    if len(environments) > 1: # there is a difference in the properties of a proxy environment and the true environment
        num_actions = len(environments[-1].reward_fns)
    else:
        num_actions = environments[-1].num_actions
    state_class = GetState(num_actions, head, state_forms=list(zip(args.state_names, args.state_forms)))
    state_class.minmax = compute_minmax(state_class, dataset_path)
    states = load_states(state_class.get_state, dataset_path)
    return states, num_actions, state_class

def random_actions(args, true_environment):
    states, num_actions, state_class = get_states(args, true_environment)
    actions = [np.random.randint(num_actions) for _ in range(args.num_stack, len(states))]
    actions = hot_actions(actions, num_actions)
    states = np.array([states[i-args.num_stack:i].flatten().tolist() for i in range(args.num_stack, len(states))])
    return np.array(actions), states, num_actions, state_class

def range_Qvals(args, true_environment, minmax):
    states, num_actions, state_class = get_states(args, true_environment)
    minr, maxr = minmax
    Qvals = [[minr + (maxr - minr) * np.random.rand()] * num_actions for _ in range(args.num_stack, len(states))]
    states = np.array([states[i-args.num_stack:i].flatten().tolist() for i in range(args.num_stack, len(states))])
    return np.array(Qvals), states, num_actions, state_class

def action_criteria(models, values, dist_entropy, action_probs, Q_vals, optimizers, true_values):
    dist_ent = -(action_probs.squeeze() * torch.log(action_probs.squeeze() + 1e-10)).sum(dim=1).mean()
    batch_mean = action_probs.squeeze().mean(dim=0)
    batch_ent = -((batch_mean + 1e-10) * torch.log(batch_mean + 1e-10)).sum()
    print(batch_ent, dist_ent, batch_mean, action_probs[0][0])
    loss = dist_ent - batch_ent
    # loss = F.binary_cross_entropy(action_probs.squeeze(), pytorch_model.wrap(desired_actions[idxes], cuda=args.cuda).squeeze())
    for optimizer in optimizers:
        optimizer.zero_grad()
    loss.backward()
    for optimizer in optimizers:
        optimizer.step()
    return loss

def Q_criteria(models, values, dist_entropy, action_probs, Q_vals, optimizers, true_values):
    # we should probably include the criteria
    loss = (Q_vals - true_values).pow(2).mean()
    print(Q_vals[0][0])
    for optimizer in optimizers:
        optimizer.zero_grad()
    loss.backward()
    for optimizer in optimizers:
        optimizer.step()
    return loss

def pretrain(args, true_environment, desired, num_actions, state_class, states, criteria):
    # args = get_args()
    # true_environment = Paddle()
    # true_environment = PaddleNoBlocks()
    dataset_path = args.record_rollouts
    changepoint_path = args.changepoint_dir
    option_chain = OptionChain(true_environment, args.changepoint_dir, args.train_edge, args)
    head, tail = get_edge(args.train_edge)
    train_models = MultiOption(1, models[args.model_form])
    print(args.state_names, args.state_forms)
    print(state_class.minmax)
    # behavior_policy = EpsilonGreedyProbs()
    fit(args, option_chain.save_dir, true_environment, train_models, state_class, desired, states, num_actions, criteria)

def fit(args, save_dir, true_environment, train_models, state_class, desired, states, num_actions, criteria):
    train_models.initialize(args, 1, state_class)
    # print(len([desired_actions[:] for i in range(len(train_models.models))]))
    print("train_models", len(train_models.models))
    desired = pytorch_model.wrap(np.stack([desired[:] for i in range(len(train_models.models))], axis=1)) # replicating for different policies
    optimizers = []
    min_batch = 10
    batch_size = 100
    for model in train_models.models:
        optimizers.append(optim.Adam(model.parameters(), args.lr, eps=args.eps, betas=args.betas, weight_decay=args.weight_decay))
    for i in range(args.num_iters):
        # idxes = np.random.choice(list(range(len(desired))), (batch_size,), replace=False)
        start = np.random.randint(len(desired))
        idxes = [(idx + start) % len(desired) for idx in range(batch_size)]
        values, dist_entropy, action_probs, Q_vals = train_models.determine_action(pytorch_model.wrap(states[idxes], cuda=args.cuda))
        if args.model_form == "population": # train the whole population, each individually
            train_models.currentModel().current_network_index = (train_models.currentModel().current_network_index + 1) % train_models.currentModel().num_population
        # print(action_probs.transpose(1,0).shape, desired[idxes].shape)
        # print(action_probs.squeeze().shape, desired.shape)
        # print(states[idxes])
        # print(action_probs,(action_probs.squeeze() * torch.log(action_probs.squeeze())).sum(dim=1).mean())
        loss = criteria(train_models, values, dist_entropy, action_probs, Q_vals, optimizers, desired)
        print("iter ", i, " at loss: ", loss.detach().cpu())
        if i % (args.num_iters // 10):
            batch_size = max(batch_size //   2, min_batch)
        if i % args.save_interval == 0 and args.save_models: # no point in saving if not training
            print("=========SAVING MODELS==========")
            train_models.save(save_dir) # TODO: implement save_options
    train_models.save(save_dir) # TODO: implement save_options