import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import sys, glob, copy, os, collections, time
import numpy as np
from ReinforcementLearning.train_rl import sample_actions
from Models.models import pytorch_model


def initialize_optimizer(args, model):
    print(args.lr)
    if args.optim == "RMSprop":
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
    def initialize(self, args, train_models):
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

    def interUpdateModel(self, step):
        # see evolutionary methods for how this is used
        # basically, updates the model at each time step if necessary (switches between population in evo methods)
        pass

    def updateModel(self):
        # TODO: unclear whether this function should actually be inside optimizer, possibly moved to a manager class
        if self.step_counter == self.num_update_model:
            self.models.option_index = np.random.randint(len(self.models.models))
            self.step_counter = 0

    def compute_weights(self, possible_indexes, rollouts, weights, reward_index):
        if len(weights) > 0:
            # allows combination of weighting schemes, by even averaging of schemes
            weighting_name = weights
            weights = torch.zeros(len(possible_indexes)) # TODO:cuda check
            if rollouts.iscuda:
                weights = weights.cuda()
            if weighting_name.find("TD") != -1: # TODO: write TD_errors_queue
                use_vals = rollouts.TD_errors_queue[reward_index].squeeze()[possible_indexes]
                weights += (use_vals + self.weighting_lambda) / (self.weighting_lambda * len(possible_indexes) + use_vals.sum())
            if weighting_name.find("return") != -1: # TODO: assumes positive return
                use_vals = rollouts.return_queue[reward_index].squeeze()[possible_indexes].abs()
                weights += ((use_vals + self.weighting_lambda) / (self.weighting_lambda * len(possible_indexes) + use_vals.sum())).pow(2)
            if weighting_name.find("recent") != -1:
                recency_weights = torch.arange(len(possible_indexes)).float() # TODO: recency can be some function, increasing
                # recency_weights = torch.exp(torch.arange(len(possible_indexes)).float() - len(possible_indexes)) # TODO: recency can be some function, increasing
                if rollouts.iscuda:
                    recency_weights= recency_weights.cuda()
                weights += ((recency_weights + self.weighting_lambda) / (self.weighting_lambda * len(possible_indexes) + recency_weights.sum())) * .1
            weights /= weights.sum()
            weights = pytorch_model.unwrap(weights.squeeze())
            # print(weights, rollouts.return_queue)
            # print(rollouts.buffer_filled, rollouts.buffer_at)
        else:
            weights = None
        return weights



    def get_rollouts_state(self, num_grad_states, rollouts, reward_index, sequential = False, last_states = False, match_option=False, weights=""):
        ''' 
        TODO: make this less bulky, since code is mostly repeated
        '''
        if num_grad_states > 0 and not last_states: # last_states true will get the last operated states
            if sequential:
                n_states = min(rollouts.buffer_filled - num_grad_states - 1, len(rollouts.state_queue) - num_grad_states - 1)
                possible_indexes = list(range(n_states))
                weights = self.compute_weights(possible_indexes, rollouts, weights, reward_index)
                seg = np.random.choice(possible_indexes, p=weights)
                grad_indexes = list(range(seg, seg + num_grad_states))
            elif match_option:
                possible_indexes = (rollouts.option_queue == reward_index).nonzero().squeeze().cpu().tolist()
                n_states = min(rollouts.buffer_filled - 1, len(rollouts.state_queue) - 1, len(possible_indexes) - 1)
                weights = self.compute_weights(possible_indexes, rollouts, weights, reward_index)
                grad_indexes = np.random.choice(possible_indexes, min(num_grad_states, n_states), replace=False, p=weights) # should probably choose from the actually valid indices...
            else:
                n_states = min(rollouts.buffer_filled - 1, len(rollouts.state_queue) - 1)
                possible_indexes = list(range(n_states))
                # print(possible_indexes)
                weights = self.compute_weights(possible_indexes, rollouts, weights, reward_index)
                # print(np.sum(weights), self.weighting_lambda)
                grad_indexes = np.random.choice(possible_indexes, min(num_grad_states, n_states), replace=False, p=weights) # should probably choose from the actually valid indices...
                # print(grad_indexes)
            # error
            state_eval = rollouts.state_queue[grad_indexes]
            next_state_eval = rollouts.state_queue[grad_indexes+1]
            current_state_eval = rollouts.current_state_queue[grad_indexes]
            next_current_state_eval = rollouts.current_state_queue[grad_indexes+1]
            q_eval = rollouts.Qvals_queue[reward_index, grad_indexes]
            next_q_eval = rollouts.Qvals_queue[reward_index, grad_indexes+1]
            action_probs_eval = rollouts.action_probs_queue[reward_index, grad_indexes]
            action_eval = rollouts.action_queue[grad_indexes]
            next_action_eval = rollouts.action_queue[grad_indexes+1]
            rollout_returns = rollouts.return_queue[reward_index, grad_indexes+1]
            next_rollout_returns = rollouts.return_queue[reward_index, grad_indexes]
            rollout_rewards = rollouts.reward_queue[reward_index, grad_indexes]
            epsilon_eval = rollouts.epsilon_queue[grad_indexes]
            self.last_selected_indexes = grad_indexes
        else:
            state_eval = rollouts.extracted_state[:-1]
            next_state_eval = rollouts.extracted_state[1:]
            current_state_eval = rollouts.current_state[:-1]
            next_current_state_eval = rollouts.current_state[1:]
            action_eval = rollouts.actions[:-1] # the last action is a phantom action from the behavior policy
            next_action_eval = rollouts.actions[1:]
            action_probs_eval = rollouts.action_probs[reward_index, :-1]
            q_eval = rollouts.Qvals[reward_index, :-1]
            next_q_eval = rollouts.Qvals[reward_index, 1:]
            rollout_returns = rollouts.returns[reward_index, :-1]
            next_rollout_returns = rollouts.returns[reward_index, 1:]
            rollout_rewards = rollouts.rewards[reward_index,:]
            epsilon_eval = rollouts.epsilon[:-1]
            self.last_selected_indexes = list(range(len(state_eval)))
        # print(epsilon_eval)
        # print(state_eval.shape, current_state_eval.shape, next_current_state_eval.shape, action_eval.shape, rollout_returns.shape, rollout_rewards.shape, next_state_eval.shape, next_rollout_returns.shape, q_eval.shape, next_q_eval.shape)
        return state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval, action_probs_eval, epsilon_eval

    def step_optimizer(self, optimizer, model, loss, RL=-1):
        '''
        steps the optimizer. This is shared between most RL algorithms, but if it is not shared, this function must be
        overriden. An RL flag is given to help alert that this is the case. Other common cases (evolutionary methods)
        should be added here
        '''
        if RL == 0:
            optimizer.zero_grad()
            (loss).backward()
            # print("gradients", model.QFunction.weight.grad)
            # print("gradients", model.action_probs.weight.grad)
            for param in model.parameters():
                if param.grad is not None: # some parts of the network may not be used
                    param.grad.data.clamp_(-1, 1)
            optimizer.step()
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
        state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval, action_probs_eval, epsilon_eval = self.get_rollouts_state(num_dist_states, rollouts, self.models.option_index, weights=args.prioritized_replay)    
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

    def trace_safety_step(self, args, train_models, rollouts):
        trace_indexes = np.choice(list(range(min(rollouts.trace_states.shape[0] * rollouts.trace_len, rollouts.trace_filled * rollouts.trace_len))), min(args.num_trace_trajectories, rollouts.trace_filled), replace = False)
        trace_states = rollouts.trace_states[trace_indexes // rollouts.trace_len, trace_indexes % rollouts.trace_len]
        trace_actions = rollouts.trace_actions[trace_indexes // rollouts.trace_len, trace_indexes % rollouts.trace_len]
        targets = torch.ones((trace_states.shape[0], rollouts.action_space)) * ((1-args.trace_conf) / (args.action_space - 1))
        targets[trace_actions] = args.trace_conf
        values, _, action_probs, q_values = train_models.determine_action(trace_states)
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
        if rollouts.cp_filled:
            # TODO: changepoint buffer must exceed args.correlate_steps, remove this
            states = torch.cat([rollouts.changepoint_queue[rollouts.changepoint_queue_len - max(args.correlate_steps - rollouts.changepoint_at,0):], rollouts.changepoint_queue[max(rollouts.changepoint_at-args.correlate_steps,0):rollouts.changepoint_at]])
            actions = torch.cat([rollouts.changepoint_action_queue[rollouts.changepoint_queue_len - max(args.correlate_steps - rollouts.changepoint_at,0):], rollouts.changepoint_action_queue[max(rollouts.changepoint_at-args.correlate_steps,0):rollouts.changepoint_at]])
        else:
            states = rollouts.changepoint_queue[max(rollouts.changepoint_at-args.correlate_steps,0):rollouts.changepoint_at]
            actions = rollouts.changepoint_action_queue[max(rollouts.changepoint_at-args.correlate_steps,0):rollouts.changepoint_at]
        if rollouts.buffer_filled == rollouts.buffer_steps:
            print(rollouts.buffer_steps - max(args.correlate_steps - rollouts.buffer_at,0), max(rollouts.buffer_at-args.correlate_steps,0),rollouts.buffer_at)
            current_states = torch.cat([rollouts.current_state_queue[rollouts.buffer_steps - max(args.correlate_steps- rollouts.buffer_at,0):], rollouts.current_state_queue[max(rollouts.buffer_at-args.correlate_steps,0):rollouts.buffer_at]])
            # current_probs = torch.cat([rollouts.action_probs_queue[train_models.option_index, rollouts.buffer_steps - max(rollouts.buffer_at-args.correlate_steps,0):], rollouts.action_queue[max(rollouts.buffer_at-args.correlate_steps,0):rollouts.buffer_at]])
        else:
            current_states = rollouts.current_state_queue[max(rollouts.buffer_at-args.correlate_steps,0):rollouts.buffer_at]
            # current_probs = rollouts.action_probs_queue[max(rollouts.buffer_at-args.correlate_steps,0):rollouts.buffer_at]
        correlate_states = states[:,-2:]# assuming traj dim of 2
        take_action_probs = torch.zeros((args.correlate_steps, rollouts.action_space)).float()
        if args.cuda:
            take_action_probs = take_action_probs.cuda()
        take_action_probs[list(range(args.correlate_steps)), actions.squeeze()] = 1.0
        correlate_state_diversity = 1.0/(((correlate_states - correlate_states.mean(dim=0))/16).pow(2).sum() + 1e-2) # diversity is 1/sigma^2
        values, dist_entropy, action_probs, q_values = train_models.determine_action(current_states)
        values, action_probs, q_values = train_models.get_action(values, action_probs, q_values)
        # print(correlate_states, correlate_state_diversity, take_action_probs * torch.log(action_probs + 1e-10))
        # loss = -(take_action_probs * torch.log(action_probs + 1e-10)).sum() * correlate_state_diversity / 100 # l1 loss
        loss = torch.exp(-(take_action_probs - action_probs).abs().sum()) * correlate_state_diversity # l1 loss
        # print(loss, correlate_states, correlate_state_diversity, take_action_probs - action_probs)
        # print(loss)
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

    def initialize(self, args, train_models):
        '''
        TODO: use arguments to define optimizer
        '''
        super().initialize(args, train_models)
        for model in train_models.models:
            self.optimizers.append(initialize_optimizer(args, model))

    def step(self, args, train_models, rollouts):
        total_loss = 0
        self.step_counter += 1
        for _ in range(args.grad_epoch):
            state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval, action_probs_eval, epsilon_eval = self.get_rollouts_state(args.num_grad_states, rollouts, self.models.option_index, weights=args.prioritized_replay)    
            values, _, action_probs, q_values = train_models.determine_action(current_state_eval)
            _, _, anp, next_q_values = train_models.determine_action(next_current_state_eval)
            _, action_probs, q_values = train_models.get_action(values, action_probs, q_values)
            _, _, next_q_values = train_models.get_action(values, anp, next_q_values)
            # q_values, next_q_values = q_values.squeeze(), next_q_values.squeeze() # TODO: current getting rid of process number, could be added in
            # print("optimization eval (cse, nse)", current_state_eval, next_current_state_eval)
            expected_qvals = (next_q_values.max(dim=1)[0].detach() * args.gamma) + rollout_rewards
            # print("q values (q, nq, eq, nmaxq, acts)", q_values, next_q_values, expected_qvals, next_q_values.max(dim=1)[0], action_eval)
            # Compute Huber loss
            # action_eval = sample_actions(q_values, deterministic=True)
            value_loss = (q_values.gather(1, action_eval) - expected_qvals).pow(2).mean()
            # print ("loss computation vl, qvs, il", value_loss, q_values.gather(1, action_eval), (q_values[list(range(len(q_values))), action_eval] - expected_qvals).pow(2))
            # print("eq, qv, vl", expected_qvals, q_values[list(range(len(q_values))), action_eval], value_loss)
            # value_loss = F.smooth_l1_loss(q_values.gather(1, action_eval).squeeze(), expected_qvals.detach())
            total_loss += value_loss
            # Optimize the model
            # print("weight update pre tm, sm", train_models.models[train_models.option_index].QFunction.weight)
            self.step_optimizer(self.optimizers[self.models.option_index], self.models.models[self.models.option_index], value_loss, RL=0)
            # print("weight update post", train_models.models[train_models.option_index].QFunction.weight)
        return total_loss/args.grad_epoch, None, None, None, None, torch.log(action_probs)

class DDPG_optimizer(LearningOptimizer):

    def initialize(self, args, train_models):
        '''
        TODO: use arguments to define optimizer
        '''
        super().initialize(args, train_models)
        for model in train_models.models:
            self.optimizers.append(initialize_optimizer(args, model))
        self.old_models = copy.deepcopy(train_models)

    def step(self, args, train_models, rollouts):
        self.step_counter += 1
        total_loss = 0
        tpl = 0
        for _ in range(args.grad_epoch):
            state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval, action_probs_eval, epsilon_eval = self.get_rollouts_state(args.num_grad_states, rollouts, self.models.option_index, weights=args.prioritized_replay)    
            _, dist_entropy, action_probs, q_values = train_models.determine_action(current_state_eval)
            _, _, anp, next_q_values = train_models.determine_action(next_current_state_eval)
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
                            q_loss * args.value_loss_coefe + policy_loss, RL=0)
            total_loss += q_loss
            tpl +=  policy_loss
            for target_param, param in zip(self.old_models.models[train_models.option_index].parameters(), train_models.currentModel().parameters()):
                    target_param.data.copy_(
                        target_param.data * (1.0 - args.target_tau) + param.data * args.target_tau
                    )
        return total_loss/args.grad_epoch, tpl/args.grad_epoch, dist_entropy, None, None, torch.log(action_probs)

class PPO_optimizer(LearningOptimizer):
    def initialize(self, args, train_models):
        super().initialize(args, train_models)
        for model in train_models.models:
            self.optimizers.append(initialize_optimizer(args, model))
        self.old_models = copy.deepcopy(train_models)

    def step(self, args, train_models, rollouts):
        self.step_counter += 1
        # self.old_models.models[train_models.option_index].load_state_dict(train_models.currentModel().state_dict())
        # self.old_models.option_index = train_models.option_index
        for _ in range(args.grad_epoch):
            state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval, action_probs_eval, epsilon_eval = self.get_rollouts_state(args.num_grad_states, rollouts, self.models.option_index, weights=args.prioritized_replay)    
            values, dist_entropy, action_probs, qvs = train_models.determine_action(current_state_eval)
            # _, _, old_action_probs, qvs = self.old_models.determine_action(current_state_eval)
            _, action_probs, qvs = train_models.get_action(values, action_probs, qvs)
            
            # print("aps", action_probs.shape, qvs.shape)
            # values, old_action_probs, _ = train_models.get_action(values, old_action_probs, qvs)
            # print(action_eval.shape, action_probs.shape, action_probs_eval.shape, epsilon_eval.shape)
            old_action_probs = correct_epsilon(action_probs_eval, epsilon_eval)
            # print("optimization eval (cse, nse, ap, lap, acts)", current_state_eval, next_current_state_eval, action_probs, old_action_probs, action_eval)
            action_log_probs, old_action_log_probs = torch.log(action_probs + 1e-10).gather(1, action_eval), torch.log(old_action_probs + 1e-10).gather(1, action_eval)
            advantages = rollout_returns.view(-1, 1) - values
            # print("returns", rollout_returns.view(-1,1), values, (advantages.std() + 1e-5))
            # print(advantages.shape)
            oa = advantages
            a = (advantages - advantages.mean())
            astd = advantages.std()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            ratio = torch.exp(action_log_probs - old_action_log_probs.detach()).squeeze()
            # adv_targ = Variable(advantages)
            # print("state", current_state_eval, "at", advantages, "r", ratio, "v", values, "rr", rollout_returns, "rew", rollout_rewards, "alp", action_log_probs, "oalp", old_action_log_probs)
            surr1 = ratio * advantages.squeeze()
            surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * advantages.squeeze()
            # print(ratio.shape, advantages.shape, -torch.min(surr1, surr2))
            action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)
            # if torch.isnan(action_loss).sum() > 0:
            #     print("NAN", rollout_returns, values, a, astd, oa)
            # print("values (alp, oalp, advantages, log ratio, ratio, surr1, surr2)", action_log_probs, old_action_log_probs, advantages, action_log_probs - old_action_log_probs.detach(), ratio, surr1, surr2)
            value_loss = (Variable(rollout_returns) - values).pow(2).mean()
            # TODO: importance sample entropy, value?
            log_output_probs = torch.log(torch.sum(action_probs, dim=0) / action_probs.size(0) + 1e-10)
            output_probs = torch.sum(action_probs, dim=0) / action_probs.size(0)
            output_entropy = -torch.sum(log_output_probs * output_probs) * args.high_entropy
            # print ("loss computation al, vl, oe", action_loss, value_loss, output_entropy)
            # output_entropy = compute_output_entropy(args, action_probs, log_output_probs)
            entropy_loss = (dist_entropy - output_entropy) #we can have two parameters
            # print("weight update pre tm, sm", train_models.models[train_models.option_index].action_probs.weight)
            # print(train_models.currentModel().conv3.weight[0])
            self.step_optimizer(self.optimizers[self.models.option_index], self.models.models[self.models.option_index],
                            value_loss * args.value_loss_coef + action_loss + entropy_loss * args.entropy_coef, RL=0)
            # print("weight update post", train_models.models[train_models.option_index].action_probs.weight)
        return value_loss, action_loss, dist_entropy, None, entropy_loss, action_log_probs

class A2C_optimizer(LearningOptimizer):
    def initialize(self, args, train_models):
        super().initialize(args, train_models)
        for model in train_models.models:
            self.optimizers.append(initialize_optimizer(args, model))

    def step(self, args, train_models, rollouts):
        self.step_counter += 1
        state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval, action_probs_eval, epsilon_eval = self.get_rollouts_state(args.num_grad_states, rollouts, self.models.option_index, weights=args.prioritized_replay)    
        values, dist_entropy, action_probs, qv = train_models.determine_action(current_state_eval)
        values, action_probs, _ = train_models.get_action(values, action_probs, qv)
        log_output_probs = torch.log(action_probs + 1e-10).gather(1, action_eval)
        output_entropy = compute_output_entropy(args, action_probs, log_output_probs)
        advantages = rollout_returns - values
        value_loss = advantages.pow(2).mean()
        action_loss = (Variable(advantages.data) * log_output_probs.squeeze()).mean()
        # print(dist_entropy, output_entropy)
        output_probs = torch.sum(action_probs, dim=0) / action_probs.size(0)
        output_entropy = -torch.sum(log_output_probs * output_probs) * args.high_entropy
        entropy_loss = (dist_entropy - output_entropy) * args.entropy_coef # TODO: find a way to use output_entropy
        # entropy_loss = dist_entropy * args.entropy_coef
        self.step_optimizer(self.optimizers[self.models.option_index], self.models.models[self.models.option_index],
                value_loss * args.value_loss_coef + action_loss + entropy_loss, RL=0)
        return value_loss, action_loss, dist_entropy, None, entropy_loss, log_output_probs

class Distributional_optimizer(LearningOptimizer):
    def initialize(self, args, train_models):
        super().initialize(args, train_models)
        for model in train_models.models:
            self.optimizers.append(initialize_optimizer(args, model))

    def step(self, args, train_models, rollouts):
        self.step_counter += 1
        for _ in range(args.grad_epoch):
            state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval, action_probs_eval, epsilon_eval = self.get_rollouts_state(args.num_grad_states, rollouts, self.models.option_index, weights=args.prioritized_replay)    
            nvalues, dist_entropy, action_probs, nQ_values = train_models.determine_action(next_current_state_eval)
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
            self.step_optimizer(self.optimizers[self.models.option_index], self.models.models[self.models.option_index], value_loss, RL=0)
            # print (value_loss)
        return value_loss, None, None, None, None, None

class PolicyGradient_optimizer(LearningOptimizer):
    def initialize(self, args, train_models):
        super().initialize(args, train_models)
        for model in train_models.models:
            self.optimizers.append(initialize_optimizer(args, model))

    def step(self, args, train_models, rollouts):
        self.step_counter += 1
        state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval, action_probs_eval, epsilon_eval = self.get_rollouts_state(args.num_grad_states, rollouts, self.models.option_index, weights=args.prioritized_replay)    
        values, dist_entropy, action_probs, qv = train_models.determine_action(current_state_eval)
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
                action_loss + entropy_loss, RL=0)
        return None, action_loss, dist_entropy, None, entropy_loss, log_output_probs


class SARSA_optimizer(LearningOptimizer):
    def initialize(self, args, train_models):
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

    def step(self, args, train_models, rollouts):
        self.step_counter += 1
            # state_eval = Variable(torch.ones(state_eval.data.shape).cuda()) # why does turning this on help tremendously?
        for _ in range(args.grad_epoch):
            state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval, action_probs_eval, epsilon_eval = self.get_rollouts_state(args.num_grad_states, rollouts, self.models.option_index, weights=args.prioritized_replay)    
            values, dist_entropy, action_probs, q_values = train_models.determine_action(current_state_eval) 
            _, _, _, next_q_values = train_models.determine_action(next_current_state_eval)
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
                    q_loss, RL=0)
            # print("weight update post", train_models.models[train_models.option_index].QFunction.weight)
        return q_loss, None, dist_entropy, None, None, None

class TabQ_optimizer(LearningOptimizer): # very similar to SARSA, and can probably be combined
    def initialize(self, args, train_models):
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

    def step(self, args, train_models, rollouts):
        self.step_counter += 1
        for _ in range(args.grad_epoch):
            state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval, action_probs_eval, epsilon_eval = self.get_rollouts_state(args.num_grad_states, rollouts, self.models.option_index, weights=args.prioritized_replay)
            # print(state_eval, rollout_rewards.squeeze())
            values, dist_entropy, action_probs, q_values = train_models.determine_action(current_state_eval) 
            values, _, _, next_q_values = train_models.determine_action(next_current_state_eval)
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

class SupervisedLearning_optimizer(LearningOptimizer):
    def initialize(self, args, train_models):
        super().initialize(args, train_models)
        for model in train_models.models:
            self.optimizers.append(initialize_optimizer(args, model))

    def step(self, args, train_models, rollouts):
        self.step_counter += 1
        current_state_eval, action_eval = self.get_trace_state(args.num_grad_states, rollouts, self.models.option_index, weights=args.prioritized_replay)    
        values, dist_entropy, action_probs, qv = train_models.determine_action(current_state_eval)
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
                action_loss + entropy_loss, RL=0)
        return None, action_loss, dist_entropy, None, entropy_loss, log_output_probs


learning_algorithms = {"DQN": DQN_optimizer, "DDPG": DDPG_optimizer, "PPO": PPO_optimizer, 
"A2C": A2C_optimizer, "SARSA": SARSA_optimizer, "TabQ":TabQ_optimizer, "PG": PolicyGradient_optimizer,
"Dist": Distributional_optimizer}
