import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import sys, glob, copy, os, collections, time
import numpy as np
from ReinforcementLearning.train_rl import sample_actions
from Models.models import pytorch_model
import cma, cv2
import time



class PopOptim():
    def __init__(self, model, args):
        self.optims = []
        for m in model.networks:
            if args.optim == "RMSprop":
                self.optims.append(optim.RMSprop(m.parameters(), args.lr, eps=args.eps, alpha=args.alpha))
            elif args.optim == "Adam":
                self.optims.append(optim.Adam(m.parameters(), args.lr, eps=args.eps, betas=args.betas, weight_decay=args.weight_decay))
            else:
                raise NotImplementedError("Unimplemented optimization")
        self.model = model

    def zero_grad(self):
        self.optims[self.model.current_network_index].zero_grad()

    def step(self):
        for param in self.model.currentModel().parameters():
            if param.grad is not None: # some parts of the network may not be used
                param.grad.data.clamp_(-1, 1)
        self.optims[self.model.current_network_index].step()

def initialize_optimizer(args, model):
    print(args.lr)
    if args.model_form == "population":
        return PopOptim(model, args)
    elif args.optim == "RMSprop":
        return optim.RMSprop(model.parameters(), args.lr, eps=args.eps, alpha=args.alpha)
    elif args.optim == "Adam":
        return optim.Adam(model.parameters(), args.lr, eps=args.eps, betas=args.betas, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError("Unimplemented optimization")

def compute_output_entropy(args, action_probs, log_output_probs):
    output_probs = torch.sum(action_probs, dim=0) / action_probs.size(0)
    output_entropy = -torch.sum(log_output_probs * output_probs) * args.high_entropy
    # if torch.isnan(output_entropy):
    #     output_entropy = 10 # filling in with a negative value
    return output_probs


class LearningOptimizer():
    def initialize(self, args, train_models, reward_classes = None):
        '''
        Currently, just assigns the models, duration (number of steps to roll out policy before optimizing) 
        and the empty optimizers, to be filled by the values. In the future it
        can do most of the shared optimization code for different optimizers
        '''
        self.models = train_models
        self.optimizers = []
        self.current_duration = args.num_steps
        self.lr = args.lr
        self.num_update_model = args.num_update_model
        self.step_counter = 0
        self.last_selected_indexes = None
        self.weighting_lambda = args.weighting_lambda # the nonzero normalization weight
        self.max_duration = self.current_duration
        self.sample_duration = args.sample_duration
        self.RL = 0
        self.dilated = False
        self.reward_swapping = args.reward_swapping
        self.max_grad_norm = 0.5 # TODO: don't hardcode

    def interUpdateModel(self, step, rewards, change, done):
        # see evolutionary methods for how this is used
        # basically, updates the model at each time step if necessary (switches between population in evo methods)
        return pytorch_model.unwrap(rewards.abs().sum()) > 0 and self.reward_swapping

    def updateModel(self, parameter):
        # TODO: unclear whether this function should actually be inside optimizer, possibly moved to a manager class
        # parameter in this case is "completed" when reward_swapping is True
        # print("switching", self.models.num_options, self.step_counter, self.num_update_model)
        if (self.step_counter == self.num_update_model and not self.reward_swapping) or (self.reward_swapping and parameter):
            self.models.option_index = np.random.randint(self.models.num_options)
            self.step_counter = 0

    def compute_weights(self, possible_indexes, rol, weights, reward_index):
        if len(weights) > 0:
            # allows combination of weighting schemes, by even averaging of schemes
            weighting_name = weights
            weights = torch.zeros(len(possible_indexes)) # TODO:cuda check
            if rol.iscuda:
                weights = weights.cuda()
            if weighting_name.find("TD") != -1: # TODO: write TD_errors_queue
                use_vals = rol.TD_errors_queue[reward_index].squeeze()[possible_indexes]
                weights += (use_vals + self.weighting_lambda) / (self.weighting_lambda * len(possible_indexes) + use_vals.sum())
            if weighting_name.find("return") != -1: # TODO: assumes positive return
                use_vals = rol.returns[reward_index].squeeze()[possible_indexes].abs()
                weights += ((use_vals + self.weighting_lambda) / (self.weighting_lambda * len(possible_indexes) + use_vals.sum())).pow(2)
            if weighting_name.find("recent") != -1:
                recency_weights = torch.arange(len(possible_indexes)).float() # TODO: recency can be some function, increasing
                # recency_weights = torch.exp(torch.arange(len(possible_indexes)).float() - len(possible_indexes)) # TODO: recency can be some function, increasing
                if rol.iscuda:
                    recency_weights= recency_weights.cuda()
                weights += ((recency_weights + self.weighting_lambda) / (self.weighting_lambda * len(possible_indexes) + recency_weights.sum())) * .1
            weights /= weights.sum()
            weights = pytorch_model.unwrap(weights.squeeze())
            # print(weights, rollouts.return_queue)
            # print(rollouts.buffer_filled, rollouts.buffer_at)
        else:
            weights = None
        return weights



    def get_rollouts_state(self, num_grad_states, rollouts, reward_index, use_range = None, sequential = False, last_states = False, match_option=False, weights="", last_buffer = False):
        ''' 
        TODO: make this less bulky, since code is mostly repeated
        '''
        if not self.dilated:
            rol = rollouts.base_rollouts
            buffer_at = rol.buffer_filled - rollouts.lag_num
        else:
            rol = rollouts.dilated_rollouts
            buffer_at = rol.buffer_filled - rollouts.lag_num
        if use_range is None:
            use_range = (0, buffer_at - int(rollouts.lag_num == 0))
        if num_grad_states > 0 and not last_states: # last_states true will get the last operated states
            if last_buffer:
                grad_indexes = list(range(use_range[0], min(max(buffer_at - num_grad_states, 0), use_range[1]), buffer_at))
                print(buffer_at, grad_indexes[:10], grad_indexes[-10:], rol.buffer_filled)
            if sequential:
                n_states = (use_range[0], min(rol.buffer_filled - num_grad_states - 1, use_range[1]))
                possible_indexes = list(range(*n_states))
                weights = self.compute_weights(possible_indexes, rol, weights, reward_index)
                seg = np.random.choice(possible_indexes, p=weights)
                grad_indexes = list(range(seg, seg + num_grad_states))
            elif match_option:
                possible_indexes = (rol.option_no == reward_index).nonzero().squeeze().cpu().tolist()
                possible_indexes = possible_indexes[use_range[0] < possible_indexes]
                possible_indexes = possible_indexes[possible_indexes < use_range[1]]
                n_states = min(rol.buffer_filled - 1, len(rol.state_queue) - 1, len(possible_indexes) - 1)
                weights = self.compute_weights(possible_indexes, rol, weights, reward_index)
                grad_indexes = np.random.choice(possible_indexes, min(num_grad_states, n_states), replace=False, p=weights) # should probably choose from the actually valid indices...
            else:
                n_states = (use_range[0], min(buffer_at, use_range[1]))

                possible_indexes = list(range(*n_states))
                # print(possible_indexes)
                weights = self.compute_weights(possible_indexes, rol, weights, reward_index)
                # print(np.sum(weights), self.weighting_lambda)
                grad_indexes = np.random.choice(possible_indexes, min(num_grad_states, len(possible_indexes)), replace=False, p=weights) # should probably choose from the actually valid indices...
                # print(grad_indexes)
            # error
            state_eval, current_state_eval, epsilon_eval, done_eval, resp_eval, action_eval, cp_states_eval, option_param_eval, option_no_eval, rollout_rewards, rollout_returns, action_probs_eval, q_eval, value_eval = rol.get_indexes(grad_indexes)
            next_state_eval, next_current_state_eval, _, _, next_resp_eval, next_action_eval, _, _, _, _, next_rollout_returns, _, next_q_eval, next_value_eval = rol.get_indexes(grad_indexes + 1)
            self.last_selected_indexes = grad_indexes
        else:
            state_eval, current_state_eval, epsilon_eval, done_eval, resp_eval, action_eval, cp_states_eval, option_param_eval, option_no_eval, rollout_rewards, rollout_returns, action_probs_eval, q_eval, value_eval = rol.get_from(0,rollouts.lag_num+1)
            next_state_eval, next_current_state_eval, _, _, next_resp_eval, next_action_eval, _, _, _, _, next_rollout_returns, _, next_q_eval, next_value_eval = rol.get_from(1,rollouts.lag_num)
            # print("states, rewards", state_eval, action_eval, rollout_rewards)
            self.last_selected_indexes = list(range(len(state_eval)))
        # print(epsilon_eval)
        # print(state_eval.shape, current_state_eval.shape, next_current_state_eval.shape, action_eval.shape, rollout_returns.shape, rollout_rewards.shape, next_state_eval.shape, next_rollout_returns.shape, q_eval.shape, next_q_eval.shape)
        return state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns[reward_index], rollout_rewards[reward_index], next_rollout_returns[reward_index], q_eval, next_q_eval, value_eval, action_probs_eval, epsilon_eval, resp_eval, rollout_returns

    def step_optimizer(self, optimizer, model, loss, RL=-1):
        '''
        steps the optimizer. This is shared between most RL algorithms, but if it is not shared, this function must be
        overriden. An RL flag is given to help alert that this is the case. Other common cases (evolutionary methods)
        should be added here
        '''
        if RL == 0:
            optimizer.zero_grad()
            # for param in model.parameters():
            #     print("bef", param.grad)
            #     break            
            (loss).backward()
            # for param in model.parameters():
            #     print("after", param.grad)
            #     break
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                         self.max_grad_norm)

            # for param in model.parameters():
            #     if param.grad is not None: # some parts of the network may not be used
            #         param.grad.data.clamp_(-1, 1)
            optimizer.step()
        elif RL == 1: # breaks abstraction, but the SARSA update is model dependent
            model.QFunction.requires_grad = False
            with torch.no_grad():
                deltas, actions, states = loss
                states = model.basis_fn(states)
                states = torch.mm(states, model.basis_matrix)
                # print (states, deltas)
                for delta, action, state in zip(deltas, actions, states):
                    model.QFunction.weight[action,:] += (self.lr * delta * state)
        elif RL == 2: # breaks abstraction, but the Tabular Q learning update is model dependent
            with torch.no_grad():
                deltas, actions, states = loss
                for delta, action, state in zip(deltas, actions, states):
                    # if len(state.shape) > 1:
                    #     state = state[0]
                    # print(state)
                    state = model.hash_function(state)
                    action = int(pytorch_model.unwrap(action))
                    if state not in model.Qtable:
                        Aprob = torch.Tensor([model.initial_aprob for _ in range(model.num_outputs)]).cuda()
                        Qval = torch.Tensor([model.initial_value for _ in range(model.num_outputs)]).cuda()
                        model.Qtable[state] = Qval
                        model.action_prob_table[state] = Aprob
                    # print(model.Qtable)
                    # print(self.lr, delta, state, model.Qtable[state][action])
                    model.Qtable[state][action] += self.lr * delta
            # print(model.name, states,actions, model.Qtable[state], deltas)
        elif RL == 3: # no update to the values, only perform backward operation
            optimizer.zero_grad()
            (loss).backward()
        else:
            raise NotImplementedError("Check that Optimization is appropriate")


    def step(self, args, train_models, rollouts):
        '''
        step the optimizer once. This computes the gradient if necessary from rollouts, and then steps the appropriate values
        in train_models. It can step either the current, or all the models.Usually accompanied with a step_optimizer
        function which steps the inner optimization algorithm
        output entropy is the batch entropy of outputs
        returns Value loss, policy loss, distribution entropy, output entropy, entropy loss, action log probabilities
        '''
        # raise NotImplementedError("step should be overriden")
        pass

    def distibutional_sparcity_step(self, args, train_models, rollouts):
        layer_values = []
        num_dist_states = args.num_grad_states * 10
        state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval, value_eval, action_probs_eval, epsilon_eval, resp_eval, full_rollout_returns = self.get_rollouts_state(num_dist_states, rollouts, self.models.option_index, weights=args.prioritized_replay)    
        layers = train_models.layers(current_state_eval)
        for optimizer, model, option_layer in zip(self.optimizers, train_models.models, layers):
            # for layer in option_layer:

            layer = option_layer[-1] # use the second to last layer, that is, before the 
            beta = layer.mean(dim=0)
            dkls = []
            rbeta = 1/torch.reciprocal(beta)
            dkl = ((rbeta - args.exp_beta * rbeta.pow(2)) * torch.clamp(torch.sign(beta - args.exp_beta), min=0.0,max=1.0)).detach()
            # print(sum(torch.clamp(torch.sign(beta - args.exp_beta), min=0.0,max=1.0)), beta, args.exp_beta)
            # print("dkl", dkl)
            optimizer.zero_grad()
            (args.dist_coef * dkl * beta).mean().backward()
            for param in model.parameters():
                if param.grad is not None: # some parts of the network may not be used
                    param.grad.data.clamp_(-1, 1)
            optimizer.step()

    def trace_safety_step(self, args, train_models, rollouts): # TODO: this does not work at the moment...
        trace_indexes = np.choice(list(range(min(rollouts.trace_states.shape[0] * rollouts.trace_len, rollouts.trace_filled * rollouts.trace_len))), min(args.num_trace_trajectories, rollouts.trace_filled), replace = False)
        trace_states = rollouts.trace_states[trace_indexes // rollouts.trace_len, trace_indexes % rollouts.trace_len]
        trace_actions = rollouts.trace_actions[trace_indexes // rollouts.trace_len, trace_indexes % rollouts.trace_len]
        targets = torch.ones((trace_states.shape[0], rollouts.action_space)) * ((1-args.trace_conf) / (args.action_space - 1))
        targets[trace_actions] = args.trace_conf
        trace_resps = None # TODO: add resps to trace
        values, _, action_probs, q_values = train_models.determine_action(trace_states, trace_resps)
        values, action_probs, q_values = train_models.get_action(values, action_probs, q_values)
        # TODO: q value regularization with traces not implemented
        # TODO: matching actions ignores other possible optimal policies
        # loss = (targets * torch.log(action_probs + 1e-10)).sum() # cross entropy loss
        # loss = (targets - action_probs).pow(2).sum() # l2 loss
        loss = (targets - action_probs).abs().sum() # l1 loss
        optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            if param.grad is not None: # some parts of the network may not be used
                param.grad.data.clamp_(-1, 1)
        optimizer.step()
        return loss

    def correlate_diversity_step(self, args, train_models, rollouts):
        '''
        Enforces diversity in the correlate state through importance sampling:
            (diversity in correlate state * 
        '''
        states, current_state, _, dones, _, actions, _, _, _, returns, _, _, _ = rollouts.buffer_get_last(0, args.correlate_steps)
        correlate_states = states[1:,-2:]# assuming traj dim of 2
        take_action_probs = torch.zeros((args.correlate_steps-1, rollouts.action_space)).float()
        noop_action_probs = torch.zeros((args.correlate_steps-1, rollouts.action_space)).float()
        if args.cuda:
            take_action_probs = take_action_probs.cuda()
            noop_action_probs = noop_action_probs.cuda()
        take_action_probs[list(range(args.correlate_steps-1)), actions.squeeze()[:-1]] = 1.0
        noop_action_probs[:,0] = 1.0 # TODO: assumes that noops are the 0th action
        correlate_state_variance = ((correlate_states - correlate_states.mean(dim=0)).pow(2).mean(dim=0)).sum() # diversity is 1/sigma^2

        # score is the exp(-z) * correlate_variance_cost
        correlate_state_score = torch.exp(-(correlate_states - correlate_states.mean(dim=0)).pow(2).sum(dim=1) / (torch.sqrt(correlate_state_variance) + 1e-10) - torch.sqrt(correlate_state_variance) + 10)
        # print("var", correlate_state_variance, correlate_states)
        # print(correlate_state_variance, -(correlate_states - correlate_states.mean(dim=0)).pow(2).sum(dim=1) / torch.sqrt(correlate_state_variance+1e-10), - torch.sqrt(correlate_state_variance) + 10)
        current_resps = None # TODO: add current_resps
        values, dist_entropy, action_probs, q_values = train_models.determine_action(current_states, current_resps)
        values, action_probs, q_values = train_models.get_action(values, action_probs, q_values)
        action_probs = action_probs[:-1]
        # print(correlate_states, correlate_state_diversity, take_action_probs * torch.log(action_probs + 1e-10))
        # loss = -(take_action_probs * torch.log(action_probs + 1e-10)).sum() * correlate_state_diversity / 100 # l1 loss
        ploss = torch.exp(-(take_action_probs - action_probs).abs().sum(dim=1)) * correlate_state_score / (args.correlate_steps-1) # l1 loss
        # print(take_action_probs - action_probs, torch.exp(-(take_action_probs - action_probs).abs().sum(dim=1)), correlate_state_score)
        loss = torch.max(ploss, torch.exp(-(noop_action_probs - action_probs).abs().sum(dim=1)) * correlate_state_score / (args.correlate_steps-1)).mean()# l1 loss
        # print(correlate_state_variance, ploss.mean(), loss.mean())
        # print(noop_action_probs)
        # print(correlate_states, correlate_state_variance, loss)
        # print(train_models.option_index)
        print("diversity_loss", pytorch_model.unwrap(loss)) 
        optimizer = self.optimizers[train_models.option_index]
        optimizer.zero_grad()
        loss.backward()
        for param in train_models.currentModel().parameters():
            if param.grad is not None: # some parts of the network may not be used
                param.grad.data.clamp_(-1, 1)
        optimizer.step()
        return loss



def correct_epsilon(action_probs, epsilons):
    '''
    assumes action_probs of shape [batch, num_actions]
    '''
    return action_probs * (1-epsilons) + (1/action_probs.shape[1]) * epsilons

class DQN_optimizer(LearningOptimizer):

    def initialize(self, args, train_models, reward_classes = None):
        '''
        TODO: use arguments to define optimizer
        '''
        super().initialize(args, train_models)
        for model in train_models.models:
            self.optimizers.append(initialize_optimizer(args, model))

    def step(self, args, train_models, rollouts, use_range=None):
        total_loss = 0
        # # self.step_counter += 1
        for _ in range(args.grad_epoch):
            state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval, value_eval, action_probs_eval, epsilon_eval, resp_eval, full_rollout_returns = self.get_rollouts_state(args.num_grad_states, rollouts, self.models.option_index, use_range=use_range, weights=args.prioritized_replay)    
            values, _, action_probs, q_values = train_models.determine_action(current_state_eval, resp_eval)
            _, log_probs, anp, next_q_values = train_models.determine_action(next_current_state_eval, resp_eval)
            _, action_probs, _, q_values = train_models.get_action(values, action_probs, log_probs, q_values)
            _, action_probs, log_probs, next_q_values = train_models.get_action(values, anp, log_probs, next_q_values)
            if len(state_eval) < 1:
                continue

            # q_values, next_q_values = q_values.squeeze(), next_q_values.squeeze() # TODO: current getting rid of process number, could be added in
            # print("optimization eval (cse, nse)", current_state_eval, next_current_state_eval)
            expected_qvals = (next_q_values.max(dim=1)[0].detach() * args.gamma) + rollout_rewards
            # print("q values (q, nq, eq, nmaxq, acts)", q_values, next_q_values, expected_qvals, next_q_values.max(dim=1)[0], action_eval)
            # Compute Huber loss
            # action_eval = sample_actions(q_values, deterministic=True)
            # print(state_eval, next_state_eval, rollout_rewards, action_eval, q_values.gather(1, action_eval))
            value_loss = (q_values.gather(1, action_eval) - expected_qvals).pow(2).mean()
            # print ("loss computation vl, qvs, il", value_loss, q_values.gather(1, action_eval), (q_values[list(range(len(q_values))), action_eval] - expected_qvals).pow(2))
            # print("eq, qv, vl", expected_qvals, q_values[list(range(len(q_values))), action_eval], value_loss)
            # value_loss = F.smooth_l1_loss(q_values.gather(1, action_eval).squeeze(), expected_qvals.detach())
            total_loss += value_loss
            # Optimize the model
            # print("weight update pre tm, sm", train_models.models[train_models.option_index].QFunction.weight)
            self.step_optimizer(self.optimizers[self.models.option_index], self.models.models[self.models.option_index], value_loss, RL=self.RL)
            # print("weight update post", train_models.models[train_models.option_index].QFunction.weight)
        return total_loss/args.grad_epoch, None, None, None, None, torch.log(action_probs)

class DDPG_optimizer(LearningOptimizer):

    def initialize(self, args, train_models, reward_classes = None):
        '''
        TODO: use arguments to define optimizer
        '''
        super().initialize(args, train_models)
        for model in train_models.models:
            self.optimizers.append(initialize_optimizer(args, model))
        self.old_models = copy.deepcopy(train_models)

    def step(self, args, train_models, rollouts, use_range=None):
        # # self.step_counter += 1
        total_loss = 0
        tpl = 0
        for _ in range(args.grad_epoch):
            state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval, value_eval, action_probs_eval, epsilon_eval, resp_eval, full_rollout_returns = self.get_rollouts_state(args.num_grad_states, rollouts, self.models.option_index, use_range=use_range, weights=args.prioritized_replay)    
            _, dist_entropy, action_probs, q_values = train_models.determine_action(current_state_eval, resp_eval)
            _, _, anp, next_q_values = train_models.determine_action(next_current_state_eval, resp_eval)
            _, action_probs, qvs = train_models.get_action(values, action_probs, q_values)
            _, _, next_q_values = train_models.get_action(values, anp, next_q_values)
            expected_qvals = (next_q_values.max(dim=1)[0] * args.gamma) + rollout_rewards
            action_eval = sample_actions(action_probs, deterministic=True)
            # Compute Huber loss
            # TODO by squeezing q values we are assuming no process number
            # TODO not using smooth l1 loss because it doesn't seem to work...
            # TODO: Policy loss does not work for discrete (probably)
            q_loss = (expected_qvals.detach() - q_values.squeeze().gather(1, action_eval)).norm().pow(2)/action_eval.size(0)
            policy_loss = -q_values.gather(1, sample_actions(action_probs, deterministic=True).unsqueeze(1)).mean()
            self.step_optimizer(self.optimizers[self.models.option_index], self.models.models[self.models.option_index],
                            q_loss * args.value_loss_coefe + policy_loss, RL=self.RL)
            total_loss += q_loss
            tpl +=  policy_loss
            for target_param, param in zip(self.old_models.models[train_models.option_index].parameters(), train_models.currentModel().parameters()):
                    target_param.data.copy_(
                        target_param.data * (1.0 - args.target_tau) + param.data * args.target_tau
                    )
        return total_loss/args.grad_epoch, tpl/args.grad_epoch, dist_entropy, None, None, torch.log(action_probs)

class PPO_optimizer(LearningOptimizer):
    def initialize(self, args, train_models, reward_classes = None):
        super().initialize(args, train_models)
        for model in train_models.models:
            self.optimizers.append(initialize_optimizer(args, model))
        # self.old_models = copy.deepcopy(train_models)

    def step(self, args, train_models, rollouts, use_range=None):
        # self.step_counter += 1
        # self.old_models.models[train_models.option_index].load_state_dict(train_models.currentModel().state_dict())
        # self.old_models.option_index = train_models.option_index
        # print("GRAD EPOCH", args.grad_epoch)
        for _ in range(args.grad_epoch):
            # s = time.time()
            state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval, value_eval, action_probs_eval, epsilon_eval, resp_eval, full_rollout_returns = self.get_rollouts_state(args.num_grad_states, rollouts, self.models.option_index, use_range=use_range, weights=args.prioritized_replay)
            # at = time.time()
            # print("lo5ading", at-s)
            if len(state_eval) <= 1: # can't take std of a single value
                value_loss, action_loss, dist_entropy, entropy_loss, action_log_probs = None, None, None, None, None
                continue
            values, log_probs, action_probs, qvs = train_models.determine_action(current_state_eval, resp_eval)
            # _, _, old_action_probs, qvs = self.old_models.determine_action(current_state_eval)
            values, action_probs, log_probs, qvs = train_models.get_action(values, action_probs, log_probs, qvs)
            # for i in range(len(state_eval)):
            #     print(rollout_returns[i], rollout_rewards[i], values[i], action_eval[i], train_models.option_index)
            #     cv2.imshow('frame',pytorch_model.unwrap(state_eval[i].view(84,84)))
            #     a = pytorch_model.unwrap(next_state_eval[i].view(84,84) - state_eval[i].view(84,84))
            #     a[a < 0] = 0
            #     cv2.imshow('frame diff',a)
            #     cv2.imshow('next frame',pytorch_model.unwrap(next_state_eval[i].view(84,84)))
            #     if cv2.waitKey(10000) & 0xFF == ord('q'):
            #         pass
            # print(train_models.option_index, rollout_rewards, rollout_returns, state_eval, action_eval, action_probs)
            # print("aps", action_probs.shape, qvs.shape)
            # values, old_action_probs, _ = train_models.get_action(values, old_action_probs, qvs)
            # print(action_eval.shape, action_probs.shape, action_probs_eval.shape, epsilon_eval.shape)
            old_action_probs = correct_epsilon(action_probs_eval, epsilon_eval)[train_models.option_index]
            # print("optimization eval (cse, nse, ap, lap, acts)", current_state_eval, next_current_state_eval, action_probs, old_action_probs, action_eval)
            # print(action_probs, action_eval, old_action_probs)
            action_log_probs, old_action_log_probs = log_probs.gather(1, action_eval), torch.log(old_action_probs + 1e-10).gather(1, action_eval)
            advantages = rollout_returns.view(-1, 1) - value_eval
            # print("returns", current_state_eval, rollout_returns, values, (advantages.std() + 1e-5), advantages)
            # print(advantages.shape)
            a = (advantages - advantages.mean())
            advantages = a / (advantages.std() + 1e-5)
            ratio = torch.exp(action_log_probs - old_action_log_probs.detach()).squeeze()
            # adv_targ = Variable(advantages)
            # print("state", current_state_eval, "at", advantages, "r", ratio, "v", values, "rr", rollout_returns, "rew", rollout_rewards, "alp", action_log_probs, "oalp", old_action_log_probs)
            surr1 = ratio * advantages.squeeze().detach()
            surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * advantages.squeeze().detach()
            # print(action_probs, log_probs, old_action_probs, surr1, surr2, ratio, advantages, rollout_returns, values)
            # print(ratio.shape, advantages.shape, -torch.min(surr1, surr2))
            action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)
            # if torch.isnan(action_loss).sum() > 0:
            #     print("NAN", rollout_returns, values, a, astd, oa)
            # print("values (alp, oalp, advantages, log ratio, ratio, surr1, surr2)", action_log_probs, old_action_log_probs, advantages, action_log_probs - old_action_log_probs.detach(), ratio, surr1, surr2)
            value_loss = (rollout_returns - values).pow(2).mean()
            # print(action_loss, a.abs().sum(), value_loss, dist_entropy)
            # TODO: importance sample entropy, value?
            log_output_probs = torch.log(torch.sum(action_probs, dim=0) / action_probs.size(0) + 1e-10)
            output_probs = torch.sum(action_probs, dim=0) / action_probs.size(0)
            output_entropy = -torch.sum(log_output_probs * output_probs) * args.high_entropy
            dist_entropy = -(log_probs * action_probs).sum(-1).mean()
            # print ("loss computation al, vl, oe", action_loss, value_loss, output_entropy)
            # output_entropy = compute_output_entropy(args, action_probs, log_output_probs)
            if args.high_entropy <= 0:
                entropy_loss = -dist_entropy * args.entropy_coef
            else:
                entropy_loss = (dist_entropy - output_entropy) * args.entropy_coef # TODO: find a way to use output_entropy
            # print("weight update pre tm, sm", train_models.models[train_models.option_index].action_probs.weight)
            # print(train_models.currentModel().conv3.weight[0])
            # print(self.models.option_index, train_models.option_index)
            # print(value_loss * args.value_loss_coef, action_loss, entropy_loss * args.entropy_coef)
            # s = time.time()
            # print("operatiing", s-at)
            self.step_optimizer(self.optimizers[self.models.option_index], self.models.models[self.models.option_index],
                            value_loss * args.value_loss_coef + action_loss + entropy_loss, RL=self.RL)
            # at = time.time()
            # print("stepping", at-s)
            # print("weight update post", train_models.models[train_models.option_index].action_probs.weight)
        return value_loss, action_loss, dist_entropy, None, entropy_loss, action_log_probs

class A2C_optimizer(LearningOptimizer):
    def initialize(self, args, train_models, reward_classes = None):
        super().initialize(args, train_models)
        for model in train_models.models:
            self.optimizers.append(initialize_optimizer(args, model))

    def step(self, args, train_models, rollouts, use_range=None):
        # self.step_counter += 1
        state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval, value_eval, action_probs_eval, epsilon_eval, resp_eval, full_rollout_returns = self.get_rollouts_state(args.num_grad_states, rollouts, self.models.option_index, use_range=use_range, weights=args.prioritized_replay)    
        if len(state_eval) <= 0:
            return None, None, None, None, None, None
        values, log_probs, action_probs, qv = train_models.determine_action(current_state_eval, resp_eval)
        values, action_probs, log_probs, _ = train_models.get_action(values, action_probs, log_probs, qv)
        # log_output_probs = torch.log(action_probs + 1e-10).gather(1, action_eval)
        log_output_probs = log_probs.gather(1, action_eval)
        # print(action_probs, log_output_probs, action_eval)
        # output_entropy = compute_output_entropy(args, action_probs, log_output_probs)
        # print(state_eval, next_state_eval, action_eval, rollout_returns, rollout_rewards, values, train_models.option_index)
        # for i in range(len(state_eval)):
        #     print(rollout_returns[i], rollout_rewards[i], values[i], action_eval[i], train_models.option_index)
        #     for j in range(args.num_stack):
        #         cv2.imshow('frame',pytorch_model.unwrap(current_state_eval[i].view(args.num_stack, 84,84)[j])/255)
        #         a = pytorch_model.unwrap(next_current_state_eval[i].view(args.num_stack, 84,84) / 255 - current_state_eval[i].view(args.num_stack, 84,84) / 255)
        #         a[a < 0] = 0
        #         cv2.imshow('frame diff',a[j])
        #         cv2.imshow('next frame',pytorch_model.unwrap(next_current_state_eval[i].view(args.num_stack, 84,84)[j])/255)
        #         if cv2.waitKey(10000) & 0xFF == ord('q'):
        #             pass
        advantages = rollout_returns.detach() - values
        value_loss = advantages.pow(2).mean()
        # print(rollout_returns.squeeze(), value_loss, values.squeeze())
        action_loss = -(advantages.detach() * log_output_probs.squeeze()).mean()
        # print(value_loss,action_loss, log_output_probs, values, advantages)
        # print(advantages, action_loss)
        # print(dist_entropy, output_entropy)
        # print((log_probs * action_probs).sum(-1))
        dist_entropy = -(log_probs * action_probs).sum(-1).mean()
        output_probs = torch.sum(action_probs, dim=0) / action_probs.size(0)
        output_entropy = -torch.sum(log_output_probs * output_probs) * args.high_entropy
        if args.high_entropy <= 0:
            entropy_loss = -dist_entropy * args.entropy_coef
        else:
            entropy_loss = (dist_entropy - output_entropy) * args.entropy_coef # TODO: find a way to use output_entropy
        # entropy_loss = dist_entropy * args.entropy_coef
        # print(values)
        # bef = self.models.models[0].linear1.weight.clone()
        self.step_optimizer(self.optimizers[self.models.option_index], self.models.models[self.models.option_index],
                value_loss * args.value_loss_coef + action_loss + entropy_loss, RL=self.RL)
        # print ("after", (bef - self.models.models[0].linear1.weight).abs().sum())
        return value_loss, action_loss, dist_entropy, None, entropy_loss, log_output_probs

class SoftActorCritic_optimizer(LearningOptimizer):
    def initialize(self, args, train_models, reward_classes = None):
        super().initialize(args, train_models)
        for model in train_models.models:
            self.optimizers.append(initialize_optimizer(args, model))

    def step(self, args, train_models, rollouts, use_range=None):
        # self.step_counter += 1
        state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval, value_eval, action_probs_eval, epsilon_eval, resp_eval, full_rollout_returns = self.get_rollouts_state(args.num_grad_states, rollouts, self.models.option_index, use_range=use_range, weights=args.prioritized_replay)    
        values, dist_entropy, action_probs, q_vals = train_models.determine_action(current_state_eval, resp_eval)
        values, action_probs, q_vals = train_models.get_action(values, action_probs, q_vals)
        log_output_probs = torch.log(action_probs + 1e-10)
        log_action_probs = log_output_probs.gather(1, action_eval)
        value_loss = (values - (q_vals - log_output_probs).mean(dim=1)).pow(2).mean()
        nvals, _, anp, next_q_values = train_models.determine_action(next_current_state_eval, resp_eval)
        next_values, _, next_q_values = train_models.get_action(nvals, anp, next_q_values)
        expected_qvals = next_values.squeeze() * args.gamma + rollout_rewards
        print((q_vals - log_output_probs).mean(dim=1), log_output_probs, values, q_vals.gather(1, action_eval), expected_qvals, rollout_rewards, next_values)
        q_loss = (q_vals.gather(1, action_eval) - expected_qvals).pow(2).mean()
        sampled_actions = torch.stack([sample_actions(prob, deterministic=False).squeeze() for prob in action_probs], dim=0).unsqueeze(1) # TODO not sure if this messes with gradients
        # print(sampled_actions, action_eval)
        action_loss = -(log_output_probs.gather(1, sampled_actions) - q_vals.gather(1, sampled_actions)).mean()
        # print(action_probs, log_output_probs, action_eval)
        # output_entropy = compute_output_entropy(args, action_probs, log_output_probs)
        # print(state_eval, next_state_eval, action_eval, rollout_returns, rollout_rewards, values, train_models.option_index)
        # advantages = rollout_returns - q_vals.gather(sampled_actions, 1)
        # value_loss = advantages.pow(2).mean()
        # action_loss = -(Variable(advantages.data) * log_output_probs.squeeze()).mean()
        # print(advantages, action_loss)
        # print(dist_entropy, output_entropy)
        # output_probs = torch.sum(action_probs, dim=0) / action_probs.size(0)
        # output_entropy = -torch.sum(log_output_probs * output_probs) * args.high_entropy
        # entropy_loss = (dist_entropy[train_models.option_index] - output_entropy) * args.entropy_coef # TODO: find a way to use output_entropy
        # entropy_loss = dist_entropy * args.entropy_coef
        self.step_optimizer(self.optimizers[self.models.option_index], self.models.models[self.models.option_index],
                value_loss * args.value_loss_coef + action_loss + q_loss * args.entropy_coef, RL=self.RL)
        return value_loss, action_loss, dist_entropy, None, q_loss, log_output_probs


class Distributional_optimizer(LearningOptimizer):
    def initialize(self, args, train_models, reward_classes = None):
        super().initialize(args, train_models)
        for model in train_models.models:
            self.optimizers.append(initialize_optimizer(args, model))

    def step(self, args, train_models, rollouts, use_range=None):
        # self.step_counter += 1
        for _ in range(args.grad_epoch):
            state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval, value_eval, action_probs_eval, epsilon_eval, resp_eval, full_rollout_returns = self.get_rollouts_state(args.num_grad_states, rollouts, self.models.option_index, use_range=use_range, weights=args.prioritized_replay)    
            nvalues, dist_entropy, action_probs, nQ_values = train_models.determine_action(next_current_state_eval, resp_eval)
            nvalues, _, nQ_values = train_models.get_action(nvalues, action_probs, nQ_values)
            # print(nQ_values)
            abest = nQ_values.max(dim=1)[1]
            # print(abest)
            m = torch.zeros(current_state_eval.shape[0], args.num_value_atoms)#[0 for i in range(args.num_value_atoms)]
            if args.cuda:
                m = m.cuda()
            p = train_models.currentModel().compute_value_distribution(next_current_state_eval)[list(range(current_state_eval.shape[0])),abest.long()]
            # print(p.shape)
            for j in range(args.num_value_atoms):
                bj = (torch.clamp(rollout_rewards + args.gamma * train_models.currentModel().value_support[j], args.value_bounds[0], args.value_bounds[1]) - args.value_bounds[0]) / train_models.currentModel().dz
                l,u = torch.floor(bj), torch.ceil(bj)
                # print(j, l,u,bj,p[:,j] * (u-bj), p[:,j] * (bj-l), p[:,j], rollout_rewards, args.gamma, args.gamma * train_models.currentModel().value_support[j])
                # print(m.shape, l.shape, p.shape, u.shape, bj.shape)
                # print(m, m[list(range(state_eval.shape[0])),l.long()], l.long(), (p[:,j] * (u-bj)).shape)
                m[list(range(state_eval.shape[0])),l.long()] += p[:,j] * (u-bj)
                m[list(range(state_eval.shape[0])),u.long()] += p[:,j] * (bj-l)
            p_current = train_models.currentModel().compute_value_distribution(current_state_eval)[list(range(current_state_eval.shape[0])),action_eval.squeeze().long()]
            # print(m, torch.log(p_current), train_models.currentModel().value_support, train_models.currentModel().dz, args.value_bounds[0])
            # print(train_models.currentModel().compute_value_distribution(current_state_eval))
            value_loss = -(m * torch.log(p_current)).sum()
            self.step_optimizer(self.optimizers[self.models.option_index], self.models.models[self.models.option_index], value_loss, RL=self.RL)
            # print (value_loss)
        return value_loss, None, None, None, None, None

class PolicyGradient_optimizer(LearningOptimizer):
    def initialize(self, args, train_models, reward_classes = None):
        super().initialize(args, train_models)
        for model in train_models.models:
            self.optimizers.append(initialize_optimizer(args, model))

    def step(self, args, train_models, rollouts, use_range=None):
        # self.step_counter += 1
        state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval, value_eval, action_probs_eval, epsilon_eval, resp_eval, full_rollout_returns = self.get_rollouts_state(args.num_grad_states, rollouts, self.models.option_index, use_range=use_range, weights=args.prioritized_replay)    
        values, dist_entropy, action_probs, qv = train_models.determine_action(current_state_eval, resp_eval)
        values, action_probs, _ = train_models.get_action(values, action_probs, qv)
        # print(state_eval, next_state_eval, rollout_rewards)
        log_output_probs = torch.log(action_probs + 1e-10).gather(1, action_eval)
        # print(torch.log(action_probs).gather(1, action_eval), torch.log(action_probs), action_eval)         
        output_entropy = compute_output_entropy(args, action_probs, log_output_probs)
        action_loss = (Variable(rollout_returns) * log_output_probs.squeeze()).mean()
        # print(dist_entropy, output_entropy, action_probs, torch.sum(action_probs, dim=0))
        entropy_loss = (dist_entropy - output_entropy) * args.entropy_coef
        # entropy_loss = dist_entropy * args.entropy_coef
        self.step_optimizer(self.optimizers[self.models.option_index], self.models.models[self.models.option_index],
                action_loss + entropy_loss, RL=self.RL)
        return None, action_loss, dist_entropy, None, entropy_loss, log_output_probs


class SARSA_optimizer(LearningOptimizer):
    def initialize(self, args, train_models, reward_classes = None):
        super().initialize(args, train_models)
        if args.optim != "base":
            for model in train_models.models:
                self.optimizers.append(initialize_optimizer(args, model))


    def step_optimizer(self, optimizer, model, loss, RL=-1):
        '''
        steps the optimizer. This is shared between most RL algorithms, but if it is not shared, this function must be
        overriden. An RL flag is given to help alert that this is the case. Other common cases (evolutionary methods)
        should be added here
        '''
        if RL == 0:
            optimizer.zero_grad()
            (loss).backward()
            # print(model.basis_matrix.grad, [ov.grad for ov in model.order_vectors], model.minmax[0], model.minmax[1])
            for param in model.parameters():
                if param.grad is not None: # some parts of the network may not be used
                    param.grad.data.clamp_(-1, 1)
            optimizer.step()
        elif RL == 1: # breaks abstraction, but the SARSA update is model dependent
            model.QFunction.requires_grad = False
            with torch.no_grad():
                deltas, actions, states = loss
                states = model.basis_fn(states)
                states = torch.mm(states, model.basis_matrix)
                # print (states, deltas)
                for delta, action, state in zip(deltas, actions, states):
                    model.QFunction.weight[action,:] += (self.lr * delta * state)
        else:
            raise NotImplementedError("Check that Optimization is appropriate")

    def step(self, args, train_models, rollouts, use_range=None):
        # self.step_counter += 1
            # state_eval = Variable(torch.ones(state_eval.data.shape).cuda()) # why does turning this on help tremendously?
        for _ in range(args.grad_epoch):
            state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval, value_eval, action_probs_eval, epsilon_eval, resp_eval, full_rollout_returns = self.get_rollouts_state(args.num_grad_states, rollouts, self.models.option_index, use_range=use_range, weights=args.prioritized_replay)    
            values, dist_entropy, action_probs, q_values = train_models.determine_action(current_state_eval, resp_eval) 
            _, _, _, next_q_values = train_models.determine_action(next_current_state_eval, resp_eval)
            _, ap, q_values = train_models.get_action(values, action_probs, q_values)
            _, ap, next_q_values = train_models.get_action(values, action_probs, next_q_values)
            # print(q_values)
            # print("rewards", rollout_rewards.shape, next_action_eval.shape, next_q_values.shape)
            expected_qvals = (next_q_values.gather(1, next_action_eval) * args.gamma).squeeze() + rollout_rewards
            # print(q_values.shape, expected_qvals.shape, action_eval.shape, expected_qvals.shape)
            delta = (expected_qvals - q_values.gather(1, action_eval).squeeze())
            q_loss = delta.pow(2).mean()
            # print("optimization eval (cse, nse)", current_state_eval, next_current_state_eval)
            # print("q values (q, nq, eq, nmaxq, acts)", q_values, next_q_values, expected_qvals, next_q_values.max(dim=1)[0], action_eval)
            # print ("loss computation vl, qvs, il", q_loss, q_values.gather(1, action_eval).squeeze(), (q_values.gather(1, action_eval).squeeze() - expected_qvals).pow(2))
            # print("weight update pre tm, sm", train_models.models[train_models.option_index].QFunction.weight)
            if args.optim == "base":
                self.step_optimizer(None, self.models.models[self.models.option_index],
                    (delta, action_eval, current_state_eval), RL=1)
            else:
                self.step_optimizer(self.optimizers[self.models.option_index], self.models.models[self.models.option_index],
                    q_loss, RL=self.RL)
            # print("weight update post", train_models.models[train_models.option_index].QFunction.weight)
        return q_loss, None, dist_entropy, None, None, None

class TabQ_optimizer(LearningOptimizer): # very similar to SARSA, and can probably be combined
    def initialize(self, args, train_models, reward_classes = None):
        super().initialize(args, train_models)
        # assumes tabular Q base models, which breaks abstraction

    def step_optimizer(self, optimizer, model, loss, RL=-1):
        '''
        steps the optimizer. This is shared between most RL algorithms, but if it is not shared, this function must be
        overriden. An RL flag is given to help alert that this is the case. Other common cases (evolutionary methods)
        should be added here
        '''
        if RL == 0:
            optimizer.zero_grad()
            (loss).backward()
            for param in model.parameters():
                if param.grad is not None: # some parts of the network may not be used
                    param.grad.data.clamp_(-1, 1)
            optimizer.step()
        elif RL == 1: # breaks abstraction, but the SARSA update is model dependent
            deltas, actions, states = loss
            states = model.transform_input(states)
            for delta, action, state in zip(deltas, actions, states):
                model.weight[action,:] += self.lr * delta * state
        elif RL == 2: # breaks abstraction, but the Tabular Q learning update is model dependent
            with torch.no_grad():
                deltas, actions, states = loss
                for delta, action, state in zip(deltas, actions, states):
                    # if len(state.shape) > 1:
                    #     state = state[0]
                    # print(state)
                    state = model.hash_function(state)
                    action = int(pytorch_model.unwrap(action))
                    if state not in model.Qtable:
                        Aprob = torch.Tensor([model.initial_aprob for _ in range(model.num_outputs)]).cuda()
                        Qval = torch.Tensor([model.initial_value for _ in range(model.num_outputs)]).cuda()
                        model.Qtable[state] = Qval
                        model.action_prob_table[state] = Aprob
                    # print(model.Qtable)
                    # print(self.lr, delta, state, model.Qtable[state][action])
                    model.Qtable[state][action] += self.lr * delta
            # print(model.name, states,actions, model.Qtable[state], deltas)
        else:
            raise NotImplementedError("Check that Optimization is appropriate")

    def step(self, args, train_models, rollouts, use_range=None):
        # self.step_counter += 1
        for _ in range(args.grad_epoch):
            state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval, value_eval, action_probs_eval, epsilon_eval, resp_eval, full_rollout_returns = self.get_rollouts_state(args.num_grad_states, rollouts, self.models.option_index, use_range=use_range, weights=args.prioritized_replay)
            # print(state_eval, rollout_rewards.squeeze())
            values, dist_entropy, action_probs, q_values = train_models.determine_action(current_state_eval, resp_eval) 
            values, _, _, next_q_values = train_models.determine_action(next_current_state_eval, next_resp_eval)
            _,  ap, q_values = train_models.get_action(values, action_probs, q_values)
            _, ap, next_q_values = train_models.get_action(values, action_probs, next_q_values)
            # print(state_eval, rollout_rewards.squeeze())
            expected_qvals = (next_q_values.max(dim=1)[0] * args.gamma) + rollout_rewards.squeeze()
            # faction_eval = sample_actions(q_values, deterministic=True) # action eval should be unchanged
            # print(faction_eval.shape, action_eval.shape)
            delta = (expected_qvals.detach() - q_values.gather(1, action_eval).squeeze())
            q_loss = delta.pow(2).mean()
            if args.optim == "base":
                self.step_optimizer(None, self.models.models[self.models.option_index],
                    (delta, action_eval, current_state_eval), RL=2)
            else:
                self.step_optimizer(self.optimizers[self.models.option_index], self.models.models[self.models.option_index],
                    q_loss, RL=0)
        return q_loss, None, dist_entropy, None, None, None

class Evolutionary_optimizer(LearningOptimizer):
    def initialize(self, args, train_models, reward_classes = None):
        super().initialize(args, train_models)
        self.gamma = args.gamma
        self.lag_num = args.lag_num
        self.variance_lr = args.variance_lr
        self.retest = args.retest
        self.reward_stopping = args.reward_stopping
        self.reward_check = args.reward_check
        self.OoO_eval = args.OoO_eval # out of order evaluation
        self.num_population = args.num_population
        self.done_swapping = args.done_swapping
        self.reset_current_duration(args.sample_duration, args.reward_check)
        self.sample_indexes = [[[] for j in range(args.num_population)] for i in range(self.models.num_options)]
        self.last_swap = 0
        self.reassess_num = args.reassess_num
        self.store_epsilons = args.optimizer_form == "EvoEps"
        if args.reassess_num > 0:
            self.reassess_pool = [] # stores the network
            self.reassess_values = [] # stores the return value
            self.reentered_list = ([], []) # stores the the index in the returns and index in the pool/values
            for _ in train_models.models:
                self.reassess_pool.append([None for i in range(args.reassess_num)])
                self.reassess_values.append([-1 for i in range(args.reassess_num)])
                self.reentered_list[0].append([])
                self.reentered_list[1].append([])

    def reset_current_duration(self, sample_duration, reward_check):
        # print(self.models.currentModel() )
        if self.done_swapping <= self.step_counter:
            sample_duration = 2000 # TODO: a hardcoded large number
        self.current_duration = sample_duration * self.models.num_options * self.models.currentModel().num_population * self.retest  // reward_check
        self.max_duration = sample_duration * self.models.num_options * self.models.currentModel().num_population * self.retest // reward_check
        self.sample_duration = sample_duration

    def interUpdateModel(self, step, rewards, change, done):
        '''
        if a reward is acquired, then switch the testing option. Each of the population has n tries
        reward of the form: [option num, batch size, 1]
        '''
        # print(self.models.currentModel().currentModel().get_parameters().abs().mean())
        duration_check = ((step - self.last_swap) % self.sample_duration == 0 and step != 0)
        early_stop = pytorch_model.unwrap(rewards.abs().sum()) > 0 and self.reward_stopping
        use_done = done and (self.done_swapping <= self.step_counter)
        # print((step - self.last_stop) % self.sample_duration)
        # late_stop = rewards.sum() > 0 and self.reward_stopping and duration_check
        # print("duration, es", duration_check, early_stop, pytorch_model.unwrap(rewards.sum()), rewards.shape, self.reward_stopping)
        if duration_check or early_stop or use_done:
            # update to next model
            ridx = step
            print(self.models.option_index, pytorch_model.unwrap(rewards.sum()) < 0, self.models.currentModel().current_network_index, duration_check, early_stop, use_done, self.last_swap, step, ridx)
            if early_stop:
                ridx = step - pytorch_model.unwrap((self.reward_check - torch.argmax(rewards.abs().sum(dim=0)))) + self.lag_num
                # print(step, ridx)#, pytorch_model.unwrap((self.reward_check - torch.argmax(rewards.abs().sum(dim=0)))), rewards, self.lag_num)
            self.sample_indexes[self.models.option_index][self.models.currentModel().current_network_index].append((self.last_swap, ridx))
            self.last_swap = step
            if self.OoO_eval:
                total_count = np.sum([np.sum([ len(poplist) for poplist in sample_lengths]) for sample_lengths in self.sample_indexes])
                if self.retest * self.num_population * self.models.num_options - total_count == 0:
                    self.last_swap = 0
                    return True
                option_weights = [(self.retest * self.num_population - np.sum([ len(poplist) for poplist in sample_lengths])) / (self.retest * self.num_population * self.models.num_options - total_count) for sample_lengths in self.sample_indexes] # total possible number
                # print(self.retest * self.num_population, np.sum([np.sum([ len(poplist) for poplist in sample_lengths]) for sample_lengths in self.sample_indexes]), self.sample_indexes, option_weights)
                self.models.option_index = np.random.choice(list(range(self.models.num_options)), p=option_weights)
            else:
                if self.models.currentModel().current_network_index == self.num_population - 1:
                    self.models.currentModel().current_network_index = 0
                    self.models.option_index += 1
                if self.models.option_index == self.models.num_options:
                    self.models.option_index = 0
                    self.last_swap = 0
                    return True
            if self.OoO_eval:
                total_pop = np.sum([len(popcount) for popcount in self.sample_indexes[self.models.option_index]])
                # print (total_pop, [(self.retest - len(popcount)) / (self.retest  * self.num_population - total_pop) for popcount in self.sample_indexes[self.models.option_index]])
                population_weights = [(self.retest - len(popcount)) / (self.retest  * self.num_population - total_pop) for popcount in self.sample_indexes[self.models.option_index]]
                self.models.currentModel().current_network_index = np.random.choice(list(range(self.models.models[0].num_population)), p=population_weights)
            else:
                self.models.currentModel().current_network_index += 1
            if self.store_epsilons:
                self.models.currentModel().main.set_parameters(self.models.currentModel().currentModel().get_parameters() + self.models.currentModel().mean.get_parameters())
        return False

    def updateModel(self, parameter):
        self.models.currentModel().current_network_index = 0
        self.sample_indexes = [[[] for j in range(self.models.models[0].num_population)] for i in range(self.models.num_options)]

    def alter_reentry_list(self, returns):
        oi = self.models.option_index
        new_values = returns[self.reentered_list[0][oi]]
        for i, val in zip(self.reentered_list[1][oi], new_values):
            old_val = self.reassess_values[oi].pop(i)
            net = self.reassess_pool[oi].pop(i)
            new_val = (val + old_val) / 2
            j = 0
            while j < len(self.reassess_values[oi]) and new_val > self.reassess_values[oi][j]:
                j += 1
            # print("reentered: ", old_val, val, new_val, j)
            self.reassess_values[oi].insert(j, new_val)
            self.reassess_pool[oi].insert(j, net)
            new_reassess = [v for v in zip(self.reassess_values[oi], self.reassess_pool[oi])]
            new_reassess.sort(key=lambda x: x[0])
            self.reassess_values[oi] = [v[0] for v in new_reassess]
            self.reassess_pool[oi] = [v[1] for v in new_reassess]
        self.reentered_list[0][oi] = list()
        self.reentered_list[1][oi] = list()

    def get_corresponding_returns(self, returns):
        # all returns of the form [num options, num population, num options] ( the return for each option)
        all_returns = []
        for i in range(self.models.num_options):
            option_returns = []
            for j in range(self.models.models[0].num_population):
                total_ret = []
                for k in range(self.models.num_options):
                    total_value = 0
                    n = 0
                    for (s,e) in self.sample_indexes[i][j]:
                        n += 1
                        # print(i,j,s,e,returns[k, s:e].sum())
                        if self.reward_stopping: # specialized stopping return
                            if returns[k, s:e+2].sum() < 0: # TODO: negative rewards not hardcoded
                                total_value += -1#returns[k, s:e+2].sum()# + (-(e-s) / self.sample_duration)
                            else:
                                # total_value += returns[k, s:e].sum() * self.sample_duration / (e-s)
                                # total_value += returns[k, s:e].sum() * (self.sample_duration - (e-s)) / self.sample_duration
                                # print(s,e, returns[k, e:e+2], (0.5 * (e-s) / (self.sample_duration)))
                                total_value += returns[k, s:e+2].sum()# * ((self.sample_duration - (e-s)) / (self.sample_duration))
                        else:
                            total_value += returns[k, s:e].sum()
                    print("total count", n)
                    total_value /= self.retest
                    if type(total_value) == float:
                        total_ret.append(total_value) #TODO HACKED SOLUTION, FIX
                    else:
                        total_ret.append(pytorch_model.unwrap(total_value).tolist())
                option_returns.append(total_ret)
            all_returns.append(option_returns)
        # print(all_returns)
        return np.array(all_returns)

    def reassess_insert(self, returns, networks):
        if self.reassess_num > 0:
            oi = self.models.option_index
            self.alter_reentry_list(returns)
            best_values = returns
            # print(self.reassess_values)
            for i in range(len(best_values)):
                print(best_values[i])
                if best_values[i] > self.reassess_values[oi][0]:
                    j = 1
                    while j < len(self.reassess_values[oi]) and best_values[i] > self.reassess_values[oi][j]:
                        j += 1
                    # print(best_values[-i], self.sample_duration)
                    self.reassess_values[oi].pop(0)
                    self.reassess_values[oi].insert(j-1, best_values[i])
                    self.reassess_pool[oi].pop(0)
                    self.reassess_pool[oi].insert(j-1, copy.deepcopy(networks[i]))
                else:
                    break
            print(self.reassess_values[oi])


    def single_option_returns(self, all_returns, oidx):
        returns = []
        for i in range(self.models.models[0].num_population):
            returns.append(all_returns[oidx, i, oidx])
        return np.array(returns)

    def single_option_best(self, all_returns, oidx):
        returns = []
        for i in range(self.models.num_options):
            for j in range(self.num_population):
                returns.append(all_returns[i, j, oidx].tolist())
        idxes = np.argsort(np.array(returns))
        # print(returns, idxes)
        idxes = idxes[-self.num_population:]
        indexes = []
        for idx in idxes:
            indexes.append([idx // self.num_population, idx % self.num_population])
        return np.array(returns)[idxes], np.array(indexes)

    def multi_option_returns(self, all_returns):
        returns = []
        for j in range(self.num_population):
            pop_returns = []
            for i in range(self.models.num_options):
                pop_returns.append(all_returns[i, j, i].tolist())
            returns.append(float(np.array(pop_returns).mean()))
        return np.array(returns)

    def step(self, args, train_models, rollouts, usebuffer =False):
        sample_duration = self.max_duration
        if not usebuffer:
            sample_duration = -1
        state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval, value_eval, action_probs_eval, epsilon_eval, resp_eval, full_rollout_returns = self.get_rollouts_state(sample_duration, rollouts, self.models.option_index, use_range=use_range, last_states=args.buffer_steps <= 0, last_buffer = True)    
        # print(rollout_returns[args.sample_duration * 0:args.sample_duration * (1)].shape, args.sample_duration)
        values, dist_entropy, action_probs, q_values = train_models.determine_action(current_state_eval, resp_eval)
        # values,  action_probs, q_values = train_models.get_action(values, action_probs, q_values) 
        state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval, value_eval, action_probs_eval, epsilon_eval, resp_eval, full_rollout_returns = self.get_rollouts_state(sample_duration, rollouts, 0, use_range=use_range, last_states=args.buffer_steps <= 0, last_buffer = True)    

        returns = torch.stack([rollout_returns[self.sample_duration * i:self.sample_duration * (i+1)].sum() for i in range(args.num_population)])
        _, best = torch.sort(returns)
        best = best[-int(args.select_ratio * train_models.currentModel().num_population):]
        if args.evo_gradient > 0:
            # normalize the returns:
            worst = torch.min(returns[best])
            if worst < 0:
                best_values = returns[best] - worst + torch.abs(torch.mean(returns[best]))
            else:
                best_values = returns[best]
            gradients = best_values / torch.sum(best_values + 1e-10)
            # print(gradients)
            gradient_layers = []
            for layer_idx in range(len(train_models.currentModel().networks[0].layers)):
                grad = torch.zeros(train_models.currentModel().networks[0].layers[layer_idx].weight.data.shape)
                if args.cuda:
                    grad = grad.cuda()
                for i, idx in enumerate(best):
                    grad += train_models.currentModel().networks[idx].layers[layer_idx].weight.data * gradients[i]
                gradient_layers.append(grad)
        vlr = 1.0
        if self.variance_lr > 0:
            params = []
            for i in range(train_models.currentModel().num_population):
                params.append(train_models.currentModel().networks[i].get_parameters())
            params = torch.stack(params)
            print(params.shape, params.var(dim=0).shape, torch.sqrt(params.var(dim=0)).mean())
            vlr = self.variance_lr * torch.exp(-torch.sqrt(params.var(dim=0)).mean() / .005)
            print("vlr", vlr)
        self.reassess_insert(returns[best], train_models.currentModel().networks[best]) # might have broken this
        print("returns, best", returns, best)
        new_networks = []
        for j, idx in enumerate(best):
            for i in range(int(1 / args.select_ratio)):
                new_network = copy.deepcopy(train_models.currentModel().networks[idx])
                if i == 0 and args.elitism:
                    pass
                elif i == 1 and args.evo_gradient > 0: # has to be at least two children
                    for layer_idx in range(len(new_network.layers)):
                        new_network.layers[layer_idx].weight.data = new_network.layers[layer_idx].weight.data + ((new_network.layers[layer_idx].weight.data - gradient_layers[layer_idx]) * args.evo_gradient).cuda()
                else:
                    if np.random.rand() < args.reentry_rate:
                        k = np.random.randint(len(self.reassess_pool[oi]))
                        if self.reassess_pool[oi][k] is not None:
                            new_network = copy.deepcopy(self.reassess_pool[oi][k])
                            self.reentered_list[0][oi].append(j * int(1 / args.select_ratio) + i)
                            self.reentered_list[1][oi].append(k)
                        else:
                            for layer_idx in range(len(new_network.layers)):
                                new_network.layers[layer_idx].weight.data = new_network.layers[layer_idx].weight.data + ((torch.rand(new_network.layers[layer_idx].weight.data.size())*2-1) * self.lr * vlr).cuda()
                    else:
                        for layer_idx in range(len(new_network.layers)):
                            new_network.layers[layer_idx].weight.data = new_network.layers[layer_idx].weight.data + ((torch.rand(new_network.layers[layer_idx].weight.data.size())*2-1) * self.lr * vlr).cuda()
                new_networks.append(new_network)
        train_models.currentModel().set_networks(new_networks)
        return None, None, dist_entropy, None, None, None

class EvolutionaryEpsilon_Optimizer(Evolutionary_optimizer):
    def initialize(self, args, train_models, reward_classes = None):
        super().initialize(args, train_models)
        self.weight_sharing = args.weight_sharing
        for i in range(len(self.models.models)):
            for i in range(train_models.currentModel().num_population):
                train_models.currentModel().networks[i].set_parameters((np.random.rand(train_models.currentModel().networks[0].count_parameters()) * 2 - 1)*args.init_var)

    def step(self, args, train_models, rollouts, use_range=None):
        sample_duration = self.max_duration
        if args.buffer_steps > 0:
            sample_duration = -1
        state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval, value_eval, action_probs_eval, epsilon_eval, resp_eval, full_rollout_returns = self.get_rollouts_state(sample_duration, rollouts, 0, use_range=use_range, last_states=args.buffer_steps <= 0, last_buffer = True)    

        all_returns = self.get_corresponding_returns(full_rollout_returns)
        print(all_returns)
        for ar in all_returns:
            print(np.sum(ar, axis=0))
        print(rollout_returns.shape)
        for midx in range(len(self.models.models)):
            train_models.option_index = midx
            returns = self.single_option_returns(all_returns, midx)
            step = returns[0] * train_models.currentModel().networks[0].get_parameters()
            for i,ret in enumerate(returns[1:]):
                step += ret * train_models.currentModel().networks[i].get_parameters()
            train_models.currentModel().mean.set_parameters(train_models.currentModel().mean.get_parameters() + self.lr * step / (len(returns) * args.init_var))
            for i in range(train_models.currentModel().num_population):
                train_models.currentModel().networks[i].set_parameters((np.random.rand(train_models.currentModel().networks[0].count_parameters()) * 2 - 1)*args.init_var)
        return None, None, None, None, None, None      

class GradientEvolution_optimizer(LearningOptimizer):
    # trains num_population models for n time steps, then chooses the ones with the highest return over the train period
    # keeps the performance of the last model as well
    def initialize(self, args, train_models, reward_classes = None):
        super().initialize(args, train_models)
        print("current model", train_models.currentModel())
        self.optimizer = learning_algorithms[args.base_learner]()
        self.optimizer.initialize(args, train_models)
        self.optimizers = self.optimizer.optimizers
        self.evo_optimizer = Evolutionary_optimizer()
        self.evo_optimizer.initialize(args, train_models)
        self.evo_optimizer.lr = args.evo_lr
        print("optimizer", self.optimizers)
        self.current_duration = args.num_steps // args.reward_check
        self.max_duration = args.sample_duration * (train_models.currentModel().num_population - 1)
        self.current_duration_at = 0
        train_models.currentModel().current_network_index = 0
        self.first = True
        self.total_returns = torch.zeros((len(train_models.currentModel().networks), ))
        if args.cuda:
            self.total_returns = self.total_returns.cuda()

    def reset_current_duration(self, sample_duration):
        self.max_duration = sample_duration * (self.models.currentModel().num_population - 1)
        self.evo_optimizer.max_duration = self.max_duration
    
    def step(self, args, train_models, rollouts, use_range=None):
        # print(self.max_duration, self.models.currentModel().num_population, self.first, self.current_duration_at, train_models.currentModel().current_network_index)
        if self.first:
            # Must use a buffer to store the values    
            states, current_state, _, _, _, actions, _, _, _, returns, _, _, _ = rollouts.buffer_get_last(0, args.sample_duration)
            self.total_returns[0].copy_(returns.sum())
            self.first = False
            rval = None, None, None, None, None, None
        if self.models.currentModel().current_network_index == 0:
            self.current_duration_at += args.num_steps # TODO: number of steps changess
            rval = None, None, None, None, None, None
        elif self.current_duration_at < self.max_duration:
            # print("stepping")
            rval = self.optimizer.step(args, train_models, rollouts)
            self.current_duration_at += args.num_steps # TODO: number of steps changess
        if train_models.currentModel().current_network_index == len(train_models.currentModel().networks) - 1 and self.current_duration_at % self.sample_duration == 0 and self.current_duration_at != 0:
            # when we have finished with the last model
            # use args.elitism to keep the best
            # print("evolution")
            rval = self.evo_optimizer.step(args, train_models, rollouts, usebuffer=True)
            self.current_duration_at = 0
        return rval


    def updateModel(self, parameter):
        # self.models.currentModel().current_network_index = (self.models.currentModel().current_network_index + 1 )  % self.models.currentModel().num_population
        # print(self.current_duration_at, self.sample_duration, self.models.currentModel().num_population, self.models.currentModel().current_network_index)
        if self.current_duration_at % self.sample_duration == 0 and self.current_duration_at != 0:
            self.models.currentModel().current_network_index += 1
            if self.models.currentModel().current_network_index % self.models.currentModel().num_population == 0:
                self.models.currentModel().current_network_index = 0
            super().updateModel(parameter)

class SteinVariational_optimizer(LearningOptimizer):
    def initialize(self, args, train_models, reward_classes = None):
        super().initialize(args, train_models)
        self.theta_shape = train_models.currentModel().parameter_vector().shape
        self.kernel = kernels[args.kernel](args, theta_shape)
        self.ialpha = 1.0 / args.stein_alpha
        self.optimizer = learning_algorithms[args.base_learner]()
        self.optimizer.initialize(args, train_models)
        self.optimizer.RL = 3 # sets the step function to only perform backward (manually apply gradient)
        self.optimizers = self.optimizer.optimizers
        self.weight_sharing = args.weight_sharing

    def compute_pairwise_distances(self):
        all_thetas = [self.models.currentModel().networks[j].get_parameters().clone().detach() for j in range(self.models.currentModel.num_population)]
        all_thetas = torch.stack(all_thetas)
        theta_distances = torch.stack([(all_thetas - theta).pow(2).sum(dim=1) for theta in all_thetas])# each distance is represented exactly twice, except diagonals
        upper_triangle = torch.stack(sum([[theta_distances[i,j] for j in range(i)] for i in range(self.models.currentModel)]))
        median, mean = upper_triangle.median(), upper_triangle.mean()
        return median, mean

    def step(self, args, train_models, rollouts, use_range=None):
        policy_grads = []
        value_loss, action_loss, dist_entropy = torch.zeros(1), torch.zeros(1), torch.zeros(1)
        for i in range(train_models.currentModel().num_population):
            train_models.currentModel().current_network_index = i
            vl, al, de, oe, el, alp = self.optimizer.step(args, train_models, rollouts)
            value_loss += vl
            action_loss += al
            dist_entropy += dl
            policy_grads.append(train_models.currentModel().get_gradients())
        value_loss, action_loss, dist_entropy = value_loss / train_models.currentModel().num_population, action_loss / train_models.currentModel().num_population, dist_entropy / train_models.currentModel().num_population
        deltas = []
        entropy_loss = torch.zeros(1)
        median, mean = self.compute_pairwise_distances()
        self.kernel.h = median.pow(2) / np.log(train_models.currentModel().num_population + 1)# assumes kernel has a bandwidth
        for i in range(train_models.currentModel().num_population):
            thetai = train_models.currentModel().networks[i].get_parameters().clone().detach()
            grad_thetai = torch.zeros(self.theta_shape)
            if args.cuda:
                grad_thetai = grad_thetai.cuda()
            for j in range(train_models.currentModel().num_population):
                thetaj = train_models.currentModel().networks[j].get_parameters().clone().detach()
                kij = self.kernel(thetai, thetaj)
                kij.backward()
                entropy_loss += kij
                grad_kij = theta_j.grad.clone()
                grad_thetai += self.ialpha * policy_grads[j] * kij + grad_kij
            deltas.append(thetai + self.lr * grad_thetai)
        for i in range(train_models.currentModel().num_population):
            train_models.currentModel().networks[i].set_parameters(deltas[i])
        entropy_loss = entropy_loss / (train_models.currentModel().num_population ** 2)
        return value_loss, action_loss, dist_entropy, None, entropy_loss, None

class CMAES_optimizer(Evolutionary_optimizer):
    def initialize(self, args, train_models, reward_classes = None):
        super().initialize(args, train_models)
        self.optimizers = []
        self.solutions = []
        self.weight_sharing = args.weight_sharing
        for i in range(len(self.models.models)):
            if args.load_weights and not args.freeze_initial: # TODO: initialize from non-population model
                train_models.models[i].expand_pop(args.num_population)
                xinit = pytorch_model.unwrap(train_models.models[i].mean.get_parameters())
                # TODO: parameter for sigma?
                sigma = args.init_var#pytorch_model.unwrap(torch.stack([train_models.models[i].networks[j].get_parameters() for j in range(train_models.models[i].num_population)]).var(dim=1).mean())
                print(xinit, sigma)
            else:
                xinit = pytorch_model.unwrap(train_models.models[i].mean.get_parameters())
                # xinit = (np.random.rand(train_models.currentModel().networks[0].count_parameters())-0.5)*args.init_var # initializes [-1,1]
                sigma = args.init_var
            cmaes_params = {"popsize": args.num_population} # might be different than the population in the model...
            cmaes = cma.CMAEvolutionStrategy(xinit, sigma, cmaes_params)
            self.optimizers.append(cmaes)
            self.solutions.append(cmaes.ask())
        for i in range(len(self.models.models)):
            self.assign_solutions(train_models, i)

    def assign_solutions(self, train_models, i):
        for j in range(train_models.models[i].num_population):
            train_models.models[i].networks[j].set_parameters(self.solutions[i][j])

    def step(self, args, train_models, rollouts, use_range=None):
        sample_duration = self.max_duration
        if args.buffer_steps > 0:
            sample_duration = -1
        state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval, value_eval, action_probs_eval, epsilon_eval, resp_eval, full_rollout_returns = self.get_rollouts_state(sample_duration, rollouts, 0, use_range=use_range, last_states=args.buffer_steps <= 0, last_buffer = True)    
        all_returns = self.get_corresponding_returns(full_rollout_returns)
        print(all_returns)
        for ar in all_returns:
            print(np.sum(ar, axis=0))
        for midx in range(len(self.models.models)):
            train_models.option_index = midx
            if args.parameterized_option > 0:
                returns = self.multi_option_returns(all_returns)
                solutions = self.solutions[train_models.option_index]
            elif self.weight_sharing > 0: # TODO: only support by CMAES at the moment
                returns, indexes = self.single_option_best(all_returns, midx)
                solutions = [] 
                for idx in indexes:
                    solutions.append(self.solutions[idx[0]][idx[1]])
                self.weight_sharing -= 1
            else:
                returns = self.single_option_returns(all_returns, midx)
                solutions = self.solutions[train_models.option_index]
            if args.weight_decay > 0:
                weight_loss = []
                for i, solution in enumerate(solutions):
                    total = np.linalg.norm(solution, ord=1)
                    num = np.sum(solution.shape)
                    loss = -total / num * .01 # average parameter weight hard coded
                    returns[i] += loss * args.weight_decay

            # returns = torch.stack([rollout_returns[self.sample_duration * i:self.sample_duration * (i+1)].sum() / self.sample_duration for i in range(args.num_population)])
            self.reassess_insert(returns, solutions)

            cmaes = self.optimizers[train_models.option_index]
            cmaes.tell(solutions, -1*returns)
            self.solutions[train_models.option_index] = cmaes.ask()
            if np.random.rand() < args.reentry_rate:
                k = np.random.randint(len(self.reassess_pool[midx]))
                j = np.random.randint(len(self.solutions[train_models.option_index]))
                if self.reassess_pool[midx][k] is not None:
                    new_network = copy.deepcopy(self.reassess_pool[midx][k])
                    self.reentered_list[0][midx].append(j)
                    self.reentered_list[1][midx].append(k)
                    self.solutions[midx][j] = new_network
            self.assign_solutions(train_models, train_models.option_index)
            best = cmaes.result[0]
            mean = cmaes.result[5]
            self.models.currentModel().best.set_parameters(best)
            self.models.currentModel().mean.set_parameters(mean)
        return None, None, None, None, None, None

class HindsightParametrizedLearning_optimizer(LearningOptimizer): # TODO: implement this
    def initialize(self, args, train_models, reward_classes=None):
        super().initialize(args, train_models)
        self.rl_optimizer = learning_algorithms[args.base_optimizer]()
        self.rl_optimizer.initialize(args, train_models)
        self.rl_optimizer.distilled = True
        self.reward_classes = reward_classes

    def interUpdateModel(self, step, rewards, change):
        return change

    def updateModel(self, possible_parameters):
        choice = np.random.randint(len(possible_parameters))
        self.models.set_parameter(possible_parameters[choice])

    def step(self, args, train_models, rollouts, use_range=None):
        # steps the RL optimizer for different targets
        ge = args.grad_epoch
        args.grad_epoch = 1
        for _ in range(ge):
            ti = []
            while len(ti) == 0:
                print(rollouts.dilation_filled)
                idx = np.random.randint(rollouts.dilation_filled - 1)
                di = rollouts.dilated_indexes[idx]
                dnxt = rollouts.dilated_indexes[idx+1]
                print(rollouts.dilated_change_indexes[:rollouts.dilation_filled], di, dnxt)
                ti = rollouts.dilated_change_indexes[di<rollouts.dilated_change_indexes]
                ti = ti[ti < dnxt].long()
            tidx = ti[np.random.randint(ti.size(0))]
            print(tidx)
            target = rollouts.dilated_change_targets[tidx]

            tiv = torch.argmin((rollouts.dilated_change_indexes - tidx).abs()) # index of dilated change indexes chosen
            tiv = rollouts.target_indexes[tiv].long() # get the index in the target queue
            print(rollouts.target_start, tiv)
            resp_eval, action_eval, cp_states_eval = rollouts.target_rollouts.get_indexes(list(range(tiv - rollouts.target_start // 2 + 1, tiv + rollouts.target_start // 2)), names=['resp', 'action', 'changepoint'])
            param_eval =  rollouts.dilated_rollouts.get_values(di.long(), names=['option_param'])
            param_eval = param_eval if np.random.randint(2) == 0 else target
            self.reward_classes[train_models.option_index].parameter = param_eval
            rewards = self.reward_classes[train_models.option_index].compute_reward(cp_states_eval, action_eval, resp_eval)
            rew = rewards.sum().unsqueeze(0).unsqueeze(0) # assumes all the reward is at that single point
            print(rew.shape, di.long() + rollouts.dilated_start-1, rollouts.dilated_start, rollouts.dilated_rollouts.buffer_filled)
            rollouts.dilated_rollouts.insert_rewards_at(args, rew, start_at = di.long() + rollouts.dilated_start-1)
            train_models.currentModel().option_value = torch.stack([param_eval for _ in range(args.num_grad_states)], dim=0)
            res = self.rl_optimizer.step(args, train_models, rollouts, use_range=(max(di-args.lookback, 0),di))
        args.grad_epoch = ge

# class SupervisedLearning_optimizer(LearningOptimizer): # TODO: implement this
#     def initialize(self, args, train_models):
#         super().initialize(args, train_models)
#         for model in train_models.models:
#             self.optimizers.append(initialize_optimizer(args, model))

#     def step(self, args, train_models, rollouts):
#         # self.step_counter += 1
#         current_state_eval, action_eval = self.get_trace_state(args.num_grad_states, rollouts, self.models.option_index, weights=args.prioritized_replay)    
#         values, dist_entropy, action_probs, qv = train_models.determine_action(current_state_eval)
#         values, action_probs, _ = train_models.get_action(values, action_probs, qv)
#         # print(state_eval, next_state_eval, rollout_rewards)
#         log_output_probs = torch.log(action_probs + 1e-10).gather(1, action_eval)
#         # print(torch.log(action_probs).gather(1, action_eval), torch.log(action_probs), action_eval)         
#         output_entropy = compute_output_entropy(args, action_probs, log_output_probs)
#         action_loss = (Variable(rollout_returns) * log_output_probs.squeeze()).mean()
#         # print(dist_entropy, output_entropy, action_probs, torch.sum(action_probs, dim=0))
#         entropy_loss = (dist_entropy - output_entropy) * args.entropy_coef
#         # entropy_loss = dist_entropy * args.entropy_coef
#         self.step_optimizer(self.optimizers[self.models.option_index], self.models.models[self.models.option_index],
#                 action_loss + entropy_loss, RL=0)
#         return None, action_loss, dist_entropy, None, entropy_loss, log_output_probs


learning_algorithms = {"DQN": DQN_optimizer, "DDPG": DDPG_optimizer, "PPO": PPO_optimizer, 
"A2C": A2C_optimizer, "SAC": SoftActorCritic_optimizer, "SARSA": SARSA_optimizer, "TabQ":TabQ_optimizer, "PG": PolicyGradient_optimizer,
"Dist": Distributional_optimizer, "Evo": Evolutionary_optimizer, "EpsEvo": EvolutionaryEpsilon_Optimizer, "GradEvo": GradientEvolution_optimizer, 
"CMAES": CMAES_optimizer, "SVPG": SteinVariational_optimizer, "Hind": HindsightParametrizedLearning_optimizer}
