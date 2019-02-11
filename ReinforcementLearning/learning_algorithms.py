import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import sys, glob, copy, os, collections, time
import numpy as np
from ReinforcementLearning.train_rl import sample_actions
from ReinforcementLearning.models import pytorch_model



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

    def interUpdateModel(self, step):
        # see evolutionary methods for how this is used
        # basically, updates the model at each time step if necessary (switches between policies in evo methods)
        pass

    def updateModel(self):
        # TODO: unclear whether this function should actually be inside optimizer, possibly moved to a manager class
        if self.step_counter == self.num_update_model:
            self.models.option_index = np.random.randint(len(self.models.models))
            self.step_counter = 0

    def get_rollouts_state(self, args, rollouts, reward_index, sequential = False, last_states = False, match_option=False):
        ''' 
        TODO: make this less bulky, since code is mostly repeated
        '''
        if args.num_grad_states > 0 and not last_states: # last_states true will get the last operated states
            if sequential:
                seg = np.random.choice(range(len(rollouts.state_queue) - args.num_grad_states))
                grad_indexes = list(range(seg, seg + args.num_grad_states))
            elif match_option:
                possible_indexes = (rollouts.option_queue == reward_index).nonzero().squeeze().cpu().tolist()
                n_states = min(rollouts.buffer_filled - 1, len(rollouts.state_queue) - 1, len(possible_indexes))
                grad_indexes = np.random.choice(possible_indexes, min(args.num_grad_states, n_states), replace=False) # should probably choose from the actually valid indices...
            else:
                n_states = min(rollouts.buffer_filled - 1, len(rollouts.state_queue) - 1)
                grad_indexes = np.random.choice(list(range(n_states)), min(args.num_grad_states, n_states), replace=False) # should probably choose from the actually valid indices...
            # error
            state_eval = Variable(rollouts.state_queue[grad_indexes])
            next_state_eval = Variable(rollouts.state_queue[grad_indexes+1])
            current_state_eval = Variable(rollouts.current_state_queue[grad_indexes])
            next_current_state_eval = Variable(rollouts.current_state_queue[grad_indexes+1])
            q_eval = Variable(rollouts.Qvals_queue[reward_index, grad_indexes])
            next_q_eval = Variable(rollouts.Qvals_queue[reward_index, grad_indexes+1])
            action_eval = Variable(rollouts.action_queue[grad_indexes])
            next_action_eval = Variable(rollouts.action_queue[grad_indexes+1])
            rollout_returns = Variable(rollouts.return_queue[reward_index, grad_indexes+1])
            next_rollout_returns = Variable(rollouts.return_queue[reward_index, grad_indexes])
            rollout_rewards = Variable(rollouts.reward_queue[reward_index, grad_indexes])
        else:
            state_eval = Variable(rollouts.extracted_state[:-1])
            next_state_eval = Variable(rollouts.extracted_state[1:])
            current_state_eval = Variable(rollouts.current_state[:-1])
            next_current_state_eval = Variable(rollouts.current_state[1:])
            action_eval = Variable(rollouts.actions[:-1]) # the last action is a phantom action from the behavior policy
            next_action_eval = Variable(rollouts.actions[1:])
            q_eval = Variable(rollouts.Qvals[reward_index, :-1])
            next_q_eval = Variable(rollouts.Qvals[reward_index, 1:])
            rollout_returns = Variable(rollouts.returns[reward_index, :-1])
            next_rollout_returns = Variable(rollouts.returns[reward_index, 1:])
            rollout_rewards = Variable(rollouts.rewards[reward_index,:])
        # print(state_eval.shape, current_state_eval.shape, next_current_state_eval.shape, action_eval.shape, rollout_returns.shape, rollout_rewards.shape, next_state_eval.shape, next_rollout_returns.shape, q_eval.shape, next_q_eval.shape)
        return state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval

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

class DQN_optimizer(LearningOptimizer):

    def initialize(self, args, train_models):
        '''
        TODO: use arguments to define optimizer
        '''
        super().initialize(args, train_models)
        for model in train_models.models:
            self.optimizers.append(optim.RMSprop(model.parameters(), args.lr, eps=args.eps, alpha=args.alpha))

    def step(self, args, train_models, rollouts):
        total_loss = 0
        self.step_counter += 1
        for _ in range(args.grad_epoch):
            state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval = self.get_rollouts_state(args, rollouts, self.models.option_index)    
            values, _, action_probs, q_values = train_models.determine_action(current_state_eval)
            _, _, anp, next_q_values = train_models.determine_action(next_current_state_eval)
            _, action_probs, q_values = train_models.get_action(values, action_probs, q_values)
            _, _, next_q_values = train_models.get_action(values, anp, next_q_values)
            # q_values, next_q_values = q_values.squeeze(), next_q_values.squeeze() # TODO: current getting rid of process number, could be added in
            expected_qvals = (next_q_values.max(dim=1)[0] * args.gamma) + rollout_rewards
            # Compute Huber loss
            action_eval = sample_actions(q_values, deterministic=True)
            value_loss = (q_values[list(range(len(q_values))), action_eval] - expected_qvals).abs().mean()
            # print("eq, qv, vl", expected_qvals, q_values[list(range(len(q_values))), action_eval], value_loss)
            # value_loss = F.smooth_l1_loss(q_values.gather(1, action_eval).squeeze(), expected_qvals.detach())
            total_loss += value_loss
            # Optimize the model
            self.step_optimizer(self.optimizers[self.models.option_index], self.models.models[self.models.option_index], value_loss, RL=0)
        return total_loss/args.grad_epoch, None, None, None, None, torch.log(action_probs)

class DDPG_optimizer(LearningOptimizer):

    def initialize(self, args, train_models):
        '''
        TODO: use arguments to define optimizer
        '''
        super().initialize(args, train_models)
        for model in train_models.models:
            self.optimizers.append(optim.RMSprop(model.parameters(), args.lr, eps=args.eps, alpha=args.alpha))
    def step(self, args, train_models, rollouts):
        self.step_counter += 1
        total_loss = 0
        tpl = 0
        for _ in range(args.grad_epoch):
            state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval = self.get_rollouts_state(args, rollouts, self.models.option_index)    
            _, dist_entropy, action_probs, q_values = train_models.determine_action(current_state_eval)
            _, _, anp, next_q_values = train_models.determine_action(next_current_state_eval)
            _, action_probs, qvs = train_models.get_action(values, action_probs, q_values)
            _, _, next_q_values = train_models.get_action(values, anp, next_q_values)
            expected_qvals = (next_q_values.max(dim=2)[0] * args.gamma) + rollout_rewards
            action_eval = sample_actions(q_values, deterministic=True)
            # Compute Huber loss
            # TODO by squeezing q values we are assuming no process number
            # TODO not using smooth l1 loss because it doesn't seem to work...
            # TODO: Policy loss does not work for discrete (probably)
            q_loss = (expected_qvals.detach() - q_values.squeeze().gather(1, action_eval)).norm() ** 2/action_eval.size(0)
            policy_loss = q_values.gather(1, sample_actions(action_probs, deterministic=True).unsqueeze(1)).mean()
            self.step_optimizer(self.optimizers[self.models.option_index], self.models.models[self.models.option_index],
                            q_loss + policy_loss, RL=0)
            total_loss += q_loss
            tpl +=  policy_loss
        return total_loss/args.grad_epoch, tpl/args.grad_epoch, dist_entropy, None, None, torch.log(action_probs)

class PPO_optimizer(LearningOptimizer):
    def initialize(self, args, train_models):
        super().initialize(args, train_models)
        for model in train_models.models:
            self.optimizers.append(optim.RMSprop(model.parameters(), args.lr, eps=args.eps, alpha=args.alpha))
        self.old_models = copy.deepcopy(train_models)

    def step(self, args, train_models, rollouts):
        self.step_counter += 1
        self.old_models.models[train_models.option_index].load_state_dict(train_models.currentModel().state_dict())
        self.old_models.option_index = train_models.option_index
        for _ in range(args.grad_epoch):
            state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval = self.get_rollouts_state(args, rollouts, self.models.option_index)    
            values, dist_entropy, action_probs, qvs = train_models.determine_action(current_state_eval)
            _, _, old_action_probs, qvs = self.old_models.determine_action(current_state_eval)
            _, action_probs, qvs = train_models.get_action(values, action_probs, qvs)
            # print("aps", action_probs.shape, qvs.shape)
            values, old_action_probs, _ = train_models.get_action(values, old_action_probs, qvs)
            action_log_probs, old_action_log_probs = torch.log(action_probs).index_select(1, action_eval.squeeze()), torch.log(old_action_probs).index_select(1, action_eval.squeeze())
            advantages = rollout_returns.view(-1, 1) - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            ratio = torch.exp(action_log_probs - old_action_log_probs.detach())
            adv_targ = Variable(advantages)
            # print(adv_targ.shape, ratio.shape)
            surr1 = ratio * adv_targ.squeeze()
            surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * adv_targ.squeeze()
            action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)
            
            value_loss = (Variable(rollout_returns) - values).pow(2).mean()
            
            output_probs = torch.sum(action_probs, dim=0) / action_probs.size(0)
            log_output_probs = torch.log(torch.sum(action_probs, dim=0) / action_probs.size(0))
            output_entropy = -torch.sum(log_output_probs * output_probs) * args.high_entropy
            entropy_loss = (dist_entropy - output_entropy) #we can have two parameters
            self.step_optimizer(self.optimizers[self.models.option_index], self.models.models[self.models.option_index],
                            value_loss * args.value_loss_coef + action_loss + entropy_loss * args.entropy_coef, RL=0)
        return value_loss, action_loss, dist_entropy, output_entropy, entropy_loss, action_log_probs

class A2C_optimizer(LearningOptimizer):
    def initialize(self, args, train_models):
        super().initialize(args, train_models)
        for model in train_models.models:
            self.optimizers.append(optim.RMSprop(model.parameters(), args.lr, eps=args.eps, alpha=args.alpha))

    def step(self, args, train_models, rollouts):
        self.step_counter += 1
        state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval = self.get_rollouts_state(args, rollouts, self.models.option_index)    
        values, dist_entropy, action_probs, qv = train_models.determine_action(current_state_eval)
        values, action_probs, _ = train_models.get_action(values, action_probs, qv)
    
        output_probs = torch.sum(action_probs, dim=0) / action_probs.size(0)
        log_output_probs = torch.log(torch.sum(action_probs, dim=0) / action_probs.size(0))
        output_entropy = -torch.sum(log_output_probs * output_probs) * args.high_entropy
        advantages = rollout_returns - values
        value_loss = advantages.pow(2).mean()
        action_loss = -(Variable(advantages.data) * log_output_probs.squeeze()).mean()
        # print(dist_entropy, output_entropy)
        entropy_loss = (dist_entropy - output_entropy) * args.entropy_coef
        self.step_optimizer(self.optimizers[self.models.option_index], self.models.models[self.models.option_index],
                value_loss * args.value_loss_coef + action_loss + entropy_loss, RL=0)
        return value_loss, action_loss, dist_entropy, output_entropy, entropy_loss, log_output_probs

class SARSA_optimizer(LearningOptimizer):
    def initialize(self, args, train_models):
        super().initialize(args, train_models)
        for model in train_models.models:
            self.optimizers.append(optim.RMSprop(model.parameters(), args.lr, eps=args.eps, alpha=args.alpha))


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
        if RL == 1: # breaks abstraction, but the SARSA update is model dependent
            deltas, actions, states = loss
            states = model.fourier_basis(states)
            for delta, action, state in zip(deltas, actions, states):
                model.QFunction.weight[action,:] += self.lr * delta * state
        else:
            raise NotImplementedError("Check that Optimization is appropriate")

    def step(self, args, train_models, rollouts):
        self.step_counter += 1
            # state_eval = Variable(torch.ones(state_eval.data.shape).cuda()) # why does turning this on help tremendously?
        state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval = self.get_rollouts_state(args, rollouts, self.models.option_index)    
        values, dist_entropy, action_probs, q_values = train_models.determine_action(current_state_eval) 
        _, _, _, next_q_values = train_models.determine_action(next_current_state_eval)
        _, ap, q_values = train_models.get_action(values, action_probs, q_values)
        _, ap, next_q_values = train_models.get_action(values, action_probs, next_q_values)
        # print("rewards", rollout_rewards.shape, next_action_eval.shape, next_q_values.shape)
        expected_qvals = (next_q_values.gather(1, next_action_eval) * args.gamma).squeeze() + rollout_rewards
        action_eval = sample_actions(q_values, deterministic=True) # action eval should be unchanged
        # print(q_values.shape, expected_qvals.shape, action_eval.shape, expected_qvals.shape)
        delta = (expected_qvals - q_values.gather(1, action_eval.unsqueeze(1)).squeeze())
        q_loss = delta.pow(2).mean()
        if args.optim == "base":
            self.step_optimizer(self.optimizers[self.models.option_index], self.models.models[self.models.option_index],
                (delta, action_eval, current_state_eval), RL=1)
        else:
            self.step_optimizer(self.optimizers[self.models.option_index], self.models.models[self.models.option_index],
                q_loss, RL=0)
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
        if RL == 1: # breaks abstraction, but the SARSA update is model dependent
            deltas, actions, states = loss
            states = model.transform_input(states)
            for delta, action, state in zip(deltas, actions, states):
                model.weight[action,:] += self.lr * delta * state
        if RL == 2: # breaks abstraction, but the Tabular Q learning update is model dependent
            deltas, actions, states = loss
            for delta, action, state in zip(deltas, actions, states):
                if len(state.shape) > 1:
                    state = state[0]
                state = tuple(int(v) for v in state)
                action = int(pytorch_model.unwrap(action))
                if state not in model.Qtable:
                    Aprob = torch.Tensor([model.initial_aprob for _ in range(model.num_outputs)]).cuda()
                    Qval = torch.Tensor([model.initial_value for _ in range(model.num_outputs)]).cuda()
                    model.Qtable[state] = Qval
                    model.action_prob_table[state] = Aprob
                # print(model.Qtable)
                model.Qtable[state][action] += self.lr * delta
            # print(model.name, states,actions, model.Qtable[state], deltas)
        else:
            raise NotImplementedError("Check that Optimization is appropriate")

    def step(self, args, train_models, rollouts):
        self.step_counter += 1
        state_eval, next_state_eval, current_state_eval, next_current_state_eval, action_eval, next_action_eval, rollout_returns, rollout_rewards, next_rollout_returns, q_eval, next_q_eval = self.get_rollouts_state(args, rollouts, self.models.option_index)
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
                (delta, action_eval, state_eval), RL=2)
        else:
            self.step_optimizer(self.optimizers[self.models.option_index], self.models.models[self.models.option_index],
                q_loss, RL=0)
        return q_loss, None, dist_entropy, None, None, None

learning_algorithms = {"DQN": DQN_optimizer, "DDPG": DDPG_optimizer, "PPO": PPO_optimizer, 
"A2C": A2C_optimizer, "SARSA": SARSA_optimizer, "TabQ":TabQ_optimizer}
