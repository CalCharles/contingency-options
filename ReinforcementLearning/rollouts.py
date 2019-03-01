import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from ReinforcementLearning.models import pytorch_model


class RolloutOptionStorage(object):
    def __init__(self, num_processes, obs_shape, action_space,
     extracted_shape, current_shape, buffer_steps, changepoint_queue_len, num_options, changepoint_shape):
        # TODO: storage does not currently support multiple processes, can be implemented
        self.num_processes = num_processes
        self.obs_shape = obs_shape
        self.action_space = action_space
        self.extracted_shape = extracted_shape
        self.current_shape = current_shape
        self.buffer_steps = buffer_steps
        self.changepoint_queue_len = changepoint_queue_len
        self.changepoint_shape = changepoint_shape
        self.buffer_filled = 0
        self.buffer_at = 0
        self.return_at = 0
        self.changepoint_at = 0
        if buffer_steps > 0:
            self.state_queue = torch.zeros(buffer_steps, *self.extracted_shape)
            self.current_state_queue = torch.zeros(buffer_steps, *self.current_shape)
            self.action_probs_queue = torch.zeros(num_options, buffer_steps, action_space)
            self.Qvals_queue = torch.zeros(num_options, buffer_steps, action_space)
            self.reward_queue = torch.zeros(num_options, buffer_steps)
            self.return_queue = torch.zeros(num_options, buffer_steps, 1)
            self.values_queue = torch.zeros(num_options, buffer_steps, 1)
            self.action_queue = torch.zeros(buffer_steps, 1).long()
            self.option_queue = torch.zeros(buffer_steps, 1).long()
            self.epsilon_queue = torch.zeros(buffer_steps, 1)
            self.state_queue.requires_grad = self.current_state_queue.requires_grad = self.action_probs_queue.requires_grad = self.Qvals_queue.requires_grad = self.reward_queue.requires_grad = self.return_queue.requires_grad = self.values_queue.requires_grad = self.action_queue.requires_grad = self.option_queue.requires_grad = self.epsilon_queue.requires_grad = False
        self.changepoint_queue = torch.zeros(changepoint_queue_len, *self.changepoint_shape)
        self.changepoint_action_queue = torch.zeros(self.changepoint_queue_len, 1).long()
        self.num_options = num_options
        self.iscuda = False
        self.set_parameters(1)
        self.last_step = 0
        self.cp_filled = False

    def set_parameters(self, num_steps):
        self.last_step = num_steps
        self.epsilon = torch.zeros(num_steps + 1, 1)
        self.extracted_state = torch.zeros(num_steps + 1, *self.extracted_shape)
        self.current_state = torch.zeros(num_steps + 1, *self.current_shape)
        self.rewards = torch.zeros(self.num_options, num_steps) # Pretend there are no other processes
        self.returns = torch.zeros(self.num_options, num_steps + 1, 1)
        self.action_probs = torch.zeros(self.num_options, num_steps + 1, self.action_space)
        self.Qvals = torch.zeros(self.num_options, num_steps + 1, self.action_space)
        self.value_preds = torch.zeros(self.num_options, num_steps + 1, 1)
        self.actions = torch.zeros(num_steps + 1, 1).long() # no other processes
        self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, *self.obs_shape) # no other processes

    def cut_current(self, num_steps):
        self.last_step = num_steps
        self.extracted_state = self.extracted_state[:num_steps + 1]
        self.current_state = self.current_state[:num_steps + 1]
        self.rewards = self.rewards[:, :num_steps] # Pretend there are no other processes
        self.returns = self.returns[:, :num_steps + 1]
        self.action_probs = self.action_probs[:, :num_steps + 1]
        self.Qvals = self.Qvals[:, :num_steps + 1]
        self.value_preds = self.value_preds[:, :num_steps + 1]
        self.actions = self.actions[:num_steps + 1] # no other processes
        self.epsilon = self.epsilon[:num_steps + 1]
        self.masks = self.masks[:num_steps + 1] # no other processes

    def cuda(self):
        # self.states = self.states.cuda()
        self.iscuda =True
        if self.buffer_steps > 0:
            self.state_queue = self.state_queue.cuda()
            self.Qvals_queue = self.Qvals_queue.cuda()
            self.action_probs_queue = self.action_probs_queue.cuda()
            self.return_queue = self.return_queue.cuda()
            self.reward_queue = self.reward_queue.cuda()
            self.action_queue = self.action_queue.cuda()
            self.values_queue = self.values_queue.cuda()
            self.option_queue = self.option_queue.cuda()
            self.current_state_queue = self.current_state_queue.cuda()
            self.epsilon_queue = self.epsilon_queue.cuda()
        if self.changepoint_queue_len > 0:
            self.changepoint_queue = self.changepoint_queue.cuda()
            self.changepoint_action_queue = self.changepoint_action_queue.cuda()
        self.current_state = self.current_state.cuda()
        self.extracted_state = self.extracted_state.cuda()
        self.action_probs = self.action_probs.cuda()
        self.Qvals = self.Qvals.cuda()
        self.rewards = self.rewards.cuda()
        self.value_preds = self.value_preds.cuda()
        self.returns = self.returns.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()
        self.epsilon = self.epsilon.cuda()

    def cpu(self):
        # self.states = self.states.cuda()
        self.iscuda = False
        if self.buffer_steps > 0:
            self.state_queue = self.state_queue.cpu()
            self.Qvals_queue = self.Qvals_queue.cpu()
            self.action_probs_queue = self.action_probs_queue.cpu()
            self.return_queue = self.return_queue.cpu()
            self.reward_queue = self.reward_queue.cpu()
            self.action_queue = self.action_queue.cpu()
            self.values_queue = self.values_queue.cpu()
            self.option_queue = self.option_queue.cpu()
            self.current_state_queue = self.current_state_queue.cpu()
            self.epsilon_queue = self.epsilon_queue.cpu()
        if self.changepoint_queue_len > 0:
            self.changepoint_queue = self.changepoint_queue.cpu()
            self.changepoint_action_queue = self.changepoint_action_queue.cpu()
        self.epsilon = self.epsilon.cpu()
        self.current_state = self.current_state.cpu()
        self.extracted_state = self.extracted_state.cpu()
        self.action_probs = self.action_probs.cpu()
        self.Qvals = self.Qvals.cpu()
        self.rewards = self.rewards.cpu()
        self.value_preds = self.value_preds.cpu()
        self.returns = self.returns.cpu()
        self.actions = self.actions.cpu()
        self.masks = self.masks.cpu()


    def insert(self, step, extracted_state, current_state, action_probs, action, q_vals, value_preds, option_no, changepoint_state, epsilon): # got rid of masks... might be useful though
        if self.buffer_steps > 0 and self.last_step != step: # using buffer and not the first step, which is a duplicate of the last step
            self.buffer_filled += int(self.buffer_filled < self.buffer_steps)
            self.state_queue[self.buffer_at].copy_(extracted_state.squeeze().detach())
            self.current_state_queue[self.buffer_at].copy_(current_state.squeeze().detach())
            self.option_queue[self.buffer_at].copy_(pytorch_model.wrap(option_no, cuda=self.iscuda))
            self.action_queue[self.buffer_at].copy_(action.squeeze().detach())
            self.epsilon_queue[self.buffer_at].copy_(epsilon.squeeze().detach())
            for oidx in range(self.num_options):
                self.values_queue[oidx, self.buffer_at].copy_(value_preds[oidx].squeeze().detach())
                self.Qvals_queue[oidx, self.buffer_at].copy_(q_vals[oidx].squeeze().detach())
                self.action_probs_queue[oidx, self.buffer_at].copy_(action_probs[oidx].squeeze().detach())
            if self.buffer_at == self.buffer_steps - 1:
                self.buffer_at = 0
            else:
                self.buffer_at += 1
        if self.changepoint_queue_len > 0:
            self.changepoint_action_queue[self.changepoint_at].copy_(action.squeeze().detach())
            self.changepoint_queue[self.changepoint_at].copy_(changepoint_state.squeeze().detach())
            if self.last_step != step:
                if self.changepoint_at == self.changepoint_queue_len - 1:
                    self.cp_filled = True
                    self.changepoint_at = 0
                else:
                    self.changepoint_at += 1
        self.extracted_state[step].copy_(extracted_state.squeeze())
        self.actions[step].copy_(action.squeeze())
        self.current_state[step].copy_(current_state.squeeze())
        self.epsilon[step].copy_(epsilon.squeeze())
        for oidx in range(self.num_options):
            self.value_preds[oidx, step].copy_(value_preds[oidx].squeeze())
            self.Qvals[oidx, step].copy_(q_vals[oidx].squeeze())
            self.action_probs[oidx, step].copy_(action_probs[oidx].squeeze())

    def insert_no_out(self, step, extracted_state, current_state, action, option_no, changepoint_state, epsilon): # got rid of masks... might be useful though
        if self.buffer_steps > 0 and self.last_step != step: # using buffer and not the first step, which is a duplicate of the last step
            self.buffer_filled += int(self.buffer_filled < self.buffer_steps)
            self.state_queue[self.buffer_at].copy_(extracted_state.squeeze().detach())
            self.current_state_queue[self.buffer_at].copy_(current_state.squeeze().detach())
            self.option_queue[self.buffer_at].copy_(pytorch_model.wrap(option_no, cuda=self.iscuda))
            self.action_queue[self.buffer_at].copy_(action.squeeze().detach())
            self.epsilon_queue[self.buffer_at].copy_(epsilon.squeeze().detach())
            if self.buffer_at == self.buffer_steps - 1:
                self.buffer_at = 0
            else:
                self.buffer_at += 1
        if self.changepoint_queue_len > 0:
            self.changepoint_action_queue[self.changepoint_at].copy_(action.squeeze().detach())
            self.changepoint_queue[self.changepoint_at].copy_(changepoint_state.squeeze().detach())
            if self.last_step != step:
                if self.changepoint_at == self.changepoint_queue_len - 1:
                    self.cp_filled = True
                    self.changepoint_at = 0
                else:
                    self.changepoint_at += 1
        self.extracted_state[step].copy_(extracted_state.squeeze())
        self.actions[step].copy_(action.squeeze())
        self.current_state[step].copy_(current_state.squeeze())
        self.epsilon[step].copy_(epsilon.squeeze())

    def insert_rewards(self, rewards):
        if self.buffer_steps > 0:
            for oidx in range(rewards.size(0)):
                for i, reward in enumerate(rewards[oidx]):
                    self.reward_queue[oidx, (self.buffer_at + i - rewards.size(1)) % self.buffer_steps].copy_(reward)
        self.rewards = rewards
        if self.iscuda:
            self.rewards = self.rewards.cuda()

    def compute_returns(self, args, next_value, segmented_duration=-1):
        gamma = args.gamma
        tau = args.tau
        return_format = args.return_enum
        for idx in range(self.num_options):
            if return_format == 1: # use_gae is True
                self.value_preds[idx, -1] = next_value
                gae = 0
                # removed masks because I'm not sure what they are for...
                for step in reversed(range(self.rewards.size(1))):
                    delta = self.rewards[idx, step] + gamma * self.value_preds[idx, step + 1] - self.value_preds[idx, step]
                    gae = delta + gamma * tau * gae
                    self.returns[idx, step] = gae + self.value_preds[idx, step]
                # print("return_queue", self.rewards)
            elif return_format == 2: # segmented returns
                for i in range(self.rewards.size(1) // segmented_duration):
                    # print(i)
                    self.returns[idx, (i+1) * segmented_duration] = 0 # last value
                    for step in reversed(range(segmented_duration)):
                        # print(self.rewards[(i * segmented_duration) + step], (i * segmented_duration) + step)
                        self.returns[idx, (i * segmented_duration) + step] = self.returns[idx, (i * segmented_duration) + step + 1] * gamma + self.rewards[idx, (i * segmented_duration) + step]
                # print(self.returns)
            else:
                self.returns[idx, -1] = 0 #next_value
                for step in reversed(range(self.rewards.size(1))):
                    self.returns[idx, step] = self.returns[idx, step + 1] * gamma + self.rewards[idx, step]
                # print(self.returns)
                if self.buffer_steps > 0:
                    for i, rew in enumerate(self.rewards[idx]):
                        # update the last ten returns
                        # print(i, 11-i)
                        update_last = 20 # imposes an artificial limit on Gamma
                        for j in range(update_last+1): #1, 2, 3, 4 ... 2, 3, 4 ... 3, 4, 
                            # print(self.returns.shape, self.rewards.shape)
                            # print(self.returns[0], self.return_queue[-6+j])
                            # print((self.return_at - j) % self.buffer_steps, self.state_queue[(self.return_at - j) % self.buffer_steps], self.return_queue[idx, (self.return_at - j) % self.buffer_steps], torch.tensor(np.power(gamma, j)).cuda() * rew)
                            self.return_queue[idx, (self.return_at - j) % self.buffer_steps] += torch.tensor(np.power(gamma,j)).cuda() * rew
                    
                    # for i, ret in enumerate(self.returns[idx]):
                    #     # print('copying', self.return_at + i)
                    #     self.return_queue[idx, (self.return_at + i) % self.buffer_steps].copy_(ret)
        # print(self.state_queue)
        if self.buffer_steps > 0:
            self.return_at = (self.return_at + self.returns.size(1)) % self.buffer_steps
    def compute_full_returns(self, next_value, gamma):
        self.returns = torch.zeros(self.num_options, self.rewards.size(1) + 1, 1).cuda()
        for idx in range(self.num_options):
            self.returns[idx, -1] = 0 #next_value
            for step in reversed(range(self.rewards.size(1))):
                self.returns[idx, step] = self.returns[idx, step + 1] * gamma + self.rewards[idx, step]
