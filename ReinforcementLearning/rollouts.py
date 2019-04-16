import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from Models.models import pytorch_model

# extracted_state, current_state, epsilon, dones, resps, actions, changepoint_states, option_param, rewards, returns, action_probs, Qvals, value_preds
class ReinforcementStorage(object):
    def __init__(self, num_options=None, save_length=None, extracted_shape=None, current_shape=None, changepoint_shape=None, action_space=None, resp_len=None, option_param_shape=None, source_rollout=None):
        if source_rollout is not None:
            self.extracted_shape = source_rollout.extracted_shape
            self.current_shape = source_rollout.current_shape
            self.resp_len = source_rollout.resp_len
            self.changepoint_shape = source_rollout.changepoint_shape
            self.action_space = source_rollout.action_space
            self.num_options = source_rollout.num_options
            self.option_param_shape = source_rollout.option_param_shape
        else:
            self.extracted_shape = extracted_shape
            self.current_shape = current_shape
            self.resp_len = resp_len
            self.changepoint_shape = changepoint_shape
            self.action_space = action_space
            self.num_options = num_options
            self.option_param_shape = option_param_shape
        self.reset_length(save_length)

    def copy_values(self, i, n, other, oi, on):
        for val, oval in zip(self.option_agnostic, other.option_agnostic):
            val[i:i+n] = oval[oi:oi+on]
        for val, oval in zip(self.option_specific, other.option_specific):
            val[:,i:i+n] = oval[:, oi:oi+on]

    def push_front(self, other, oi, on):
        if self.buffer_filled + on > self.buffer_steps:
            roll_num = max(self.buffer_filled - self.buffer_steps + on, 0)
            for i in range(len(self.option_agnostic)):
                self.option_agnostic[i] = self.option_agnostic[i].roll(roll_num, dim=0)
            for i in range(len(self.option_specific)):
                self.option_specific[i] = self.option_specific[i].roll(roll_num, dim=1)
        self.reset_values()
        self.copy_values(self.buffer_filled - on, on, other, oi, on)

    def copy_first(self, other, n):
        self.copy_values(self.buffer_filled - n, n, other, other.buffer_filled-n, n)

    def copy_front(self, other, oi, on):
        self.copy_values(self.buffer_filled - on, on, other, oi, on)

    def reset_lists(self):
        self.option_agnostic = [self.extracted_state, self.current_state, self.epsilon, self.dones, self.resps, self.actions, self.changepoint_states, self.option_param, self.option_num]
        self.option_specific = [self.rewards, self.returns, self.action_probs, self.Qvals, self.value_preds]
        self.name_dict = {'state': self.extracted_state, 'current_state': self.current_state, 'epsilon': self.epsilon, 'done': self.dones, 'resp': self.resps, 'actions': self.actions,
                          'changepoint': self.changepoint_states, 'option_param': self.option_param, 'option': self.option_num,
                          'rewards': self.rewards, 'returns': self.returns, 'action_probs': self.action_probs, 'Qvals': self.Qvals, 'value_preds': self.value_preds}

    def reset_values(self):
        self.extracted_state, self.current_state, self.epsilon, self.dones, self.resps, self.actions, self.changepoint_states, self.option_param, self.option_num = tuple(self.option_agnostic)
        self.rewards, self.returns, self.action_probs, self.Qvals, self.value_preds = tuple(self.option_specific)
        self.name_dict = {'state': self.extracted_state, 'current_state': self.current_state, 'epsilon': self.epsilon, 'done': self.dones, 'resp': self.resps, 'actions': self.actions,
                          'changepoint': self.changepoint_states, 'option_param': self.option_param, 'option': self.option_num,
                          'rewards': self.rewards, 'returns': self.returns, 'action_probs': self.action_probs, 'Qvals': self.Qvals, 'value_preds': self.value_preds}

    def reset_length(self, length):
        self.extracted_state = torch.zeros(save_length, *self.extracted_shape).detach()
        self.current_state = torch.zeros(save_length, *self.current_shape).detach()
        self.epsilon = torch.zeros(save_length, 1).detach()
        self.dones = torch.zeros(save_length, 1).detach()

        self.resps = torch.zeros(save_length, self.resp_len).detach().long()
        self.actions = torch.zeros(save_length, 1).detach().long()
        self.changepoint_states = torch.zeros(save_length, *self.changepoint_shape)
        self.option_param = torch.zeros(save_length, *self.option_param_shape)
        self.option_num = torch.zeros(save_length, 1)

        self.rewards = torch.zeros(self.num_options, save_length).detach()
        self.returns = torch.zeros(self.num_options, save_length, 1).detach()
        self.action_probs = torch.zeros(self.num_options, save_length, self.action_space).detach()
        self.Qvals = torch.zeros(self.num_options, save_length, self.action_space).detach()
        self.value_preds = torch.zeros(self.num_options, save_length, 1).detach()
        self.reset_lists()
        self.buffer_steps = save_length
        self.buffer_filled = 0
        for ov in self.option_agnostic + self.option_specific:
            ov.requires_grad = False


    def cuda(self):
        self.iscuda = True
        for i in range(len(self.option_agnostic)):
            self.option_agnostic[i] = self.option_agnostic[i].cuda()
        for i in range(len(self.option_specific)):
            self.option_agnostic[i] = self.option_agnostic[i].cuda()
        self.reset_values()

    def cpu(self):
        self.iscuda = False
        for i in range(len(self.option_agnostic)):
            self.option_agnostic[i] = self.option_agnostic[i].cpu()
        for i in range(len(self.option_specific)):
            self.option_agnostic[i] = self.option_agnostic[i].cpu()
        self.reset_values()

    def insert_other(self, other, oi):
        self.insert(self, False, *other.get_values(oi))

    def insert(self, reenter, extracted_state, current_state, epsilon, done, resps, action, changepoint_state, option_param, option_no, rewards=None, returns=None, action_probs=None, Qvals=None, value_preds=None, option_agnostic = False):
        if not reenter:
            if self.buffer_filled == self.buffer_steps:
                for i in range(len(self.option_agnostic)):
                    self.option_agnostic[i] = self.option_agnostic[i].roll(1, dim=0)
                for i in range(len(self.option_specific)):
                    self.option_specific[i] = self.option_specific[i].roll(1, dim=1)
                self.reset_values()
        else:
            self.buffer_filled -= 1 # if reentering, subtract 1 so that we insert to the same location. Don't reenter at very first
        self.buffer_filled += int(self.buffer_filled < self.buffer_steps)
        self.state[self.buffer_filled - 1].copy_(extracted_state.squeeze().detach())
        self.changepoint_states[self.buffer_filled - 1].copy_(changepoint_state.squeeze().detach())
        self.current_state[self.buffer_filled - 1].copy_(current_state.squeeze().detach())
        self.extracted_state[self.buffer_filled - 1].copy_(extracted_state.squeeze().detach())
        self.option_param[self.buffer_filled - 1].copy_(option_param.squeeze().detach())
        self.option_no[self.buffer_filled - 1].copy_(pytorch_model.wrap(option_no, cuda=self.iscuda))
        self.action[self.buffer_filled - 1].copy_(action.squeeze().detach())
        self.epsilon[self.buffer_filled - 1].copy_(epsilon.squeeze().detach())
        self.done[self.buffer_filled - 1].copy_(pytorch_model.wrap(int(done), cuda=self.iscuda))
        self.resp[self.buffer_filled - 1].copy_(resp.squeeze().detach())
        if not option_agnostic:
            for oidx in range(self.num_options):
                self.values[oidx, self.buffer_filled - 1].copy_(value_preds[oidx].squeeze().detach())
                self.Qvals[oidx, self.buffer_filled - 1].copy_(q_vals[oidx].squeeze().detach())
                self.action_probs[oidx, self.buffer_filled - 1].copy_(action_probs[oidx].squeeze().detach())
                if rewards is not None:
                    self.rewards[oidx, self.buffer_filled - 1].copy_(rewards[oidx].squeeze().detach())
                if returns is not None:
                    self.returns[oidx, self.buffer_filled - 1].copy_(returns[oidx].squeeze().detach())

    def insert_rewards(self, args, rewards, start_at):
        self.insert_rewards_at(args, rewards, start_at = self.buffer_filled)

    def insert_rewards_at(self, args, rewards, start_at):
        if self.buffer_filled + rewards.size(1) > self.buffer_steps:
            roll_num = max(start_at - self.buffer_steps + rewards.size(1), 0)
            self.returns = self.returns.roll(roll_num, dim=1)
        self.reward_queue[:, max(start_at - rewards.size(1), 0):start_at] = rewards.detach()
        self.compute_returns(args, rewards, self.values_queue[start_at-1], start_at - 1)
        self.reset_lists()

    def buffer_get_last(self, i, n):
        # TODO: choose values to get from the queue
        # TODO: add responsibility
        # i is the starting index (0 from the most recent)
        # n is the number taken
        # changepoint is getting from the changepoint queue
        full_state = [ov[max(self.buffer_filled - i - n,0):self.buffer_filled - i] for ov in self.option_agnostic]
        full_state += [ov[:, max(self.buffer_filled - i - n,0):self.buffer_filled - i] for ov in self.option_specific]
        return tuple(full_state)

    def get_values(self, i, names=[]):
        if len(names) > 0:
            res = []
            for n in names:
                if self.name_dict[n] in self.option_agnostic:
                    res.append(self.name_dict[n][i])
                else:
                    res.append(self.name_dict[n][:,i])
            return tuple(res)
        else:
            return tuple([oa[i] for oa in self.option_agnostic] + [os[:,i] for os in self.option_specific])

    def get_indexes(self, idxes):
        if len(names) > 0:
            res = []
            for n in names:
                if self.name_dict[n] in self.option_agnostic:
                    res.append(self.name_dict[n][idxes])
                else:
                    res.append(self.name_dict[n][:,idxes])
            return tuple(res)
        else:
            return tuple([oa[idxes] for oa in self.option_agnostic] + [os[:,idxes] for os in self.option_specific])

    def get_from(self, i, j): # i from the beginning, j from the end
        if len(names) > 0:
            res = []
            for n in names:
                if self.name_dict[n] in self.option_agnostic:
                    res.append(self.name_dict[n][i:self.buffer_filled-j])
                else:
                    res.append(self.name_dict[n][:,i:self.buffer_filled-j])
            return tuple(res)
        else:
            return tuple([oa[i:self.buffer_filled-j] for oa in self.option_agnostic] + [os[:,i:self.buffer_filled-j] for os in self.option_specific])


    def compute_returns(self, args, rewards, next_value, start_at):
        gamma = args.gamma
        tau = args.tau
        return_format = args.return_enum
        if start_at + rewards.size(1) > self.buffer_steps:
            roll_num = max(start_at - self.buffer_steps + rewards.size(1), 0)
            self.returns = self.returns.roll(roll_num, dim=1)
        # must call reset_lists afterwards

        for idx in range(self.num_options):
            update_last = min(args.buffer_clip * self.gamma_dilation, start_at * self.gamma_dilation) # imposes an artificial limit on Gamma
            if self.iscuda:
                last_values = (torch.arange(start=update_last-1, end = -1, step=-1).float() * torch.zeros(self.gamma_dilation, update_last)).t().flatten().cuda().detach()
            else:
                last_values = (torch.arange(start=update_last-1, end = -1, step=-1).float() * torch.zeros(self.gamma_dilation, update_last)).t().flatten().detach()
            for i, rew in enumerate(reversed(rewards[idx])):
                # update the last ten returns
                # print(i, 11-i)
                i = rewards.size(1) - i
                self.returns[start_at-update_last-i:start_at-i] += (torch.pow(gamma,last_values) * rew).unsqueeze(1)


# TODO: clean up the rollouts
class RolloutOptionStorage(object):
    def __init__(self, num_processes, obs_shape, action_space, resp_len,
     extracted_shape, current_shape, buffer_steps, changepoint_queue_len,
     trace_len, trace_queue_len, dilated_start, target_start, dilation_queue_len, option_param_shape,
     num_options, changepoint_shape, lag_num, cuda):
        # TODO: storage does not currently support multiple processes, can be implemented
        self.num_processes = num_processes
        self.obs_shape = obs_shape
        self.action_space = action_space
        self.extracted_shape = extracted_shape
        self.current_shape = current_shape
        self.buffer_steps = buffer_steps
        self.changepoint_queue_len = changepoint_queue_len
        self.changepoint_shape = changepoint_shape
        self.lag_num = lag_num
        self.resp_len = resp_len
        self.trace_len = trace_len
        self.trace_queue_len = trace_queue_len
        self.lag_num = lag_num
        self.num_options = num_options
        self.iscuda = False
        self.set_parameters(1)
        self.last_step = 0
        self.cp_filled = False
        self.dilated_start = dilated_start
        self.dilated_indexes = torch.zeros(dilation_queue_len * dilated_start)
        self.dilated_counter = 0
        self.dilation_filled = 0
        self.target_start = target_start
        self.dilated_change_targets = torch.zeros(dilation_queue_len, *option_param_shape)
        self.dilated_change_indexes = torch.zeros(dilation_queue_len)
        self.change_filled = 0
        self.target_counter = 0

        self.buffer_at = 0

        if buffer_steps < 0:
            buffer_steps = 1
        self.base_rollouts = ReinforcementStorage(num_options, buffer_steps, extracted_shape, current_shape, changepoint_shape, action_space, resp_len, option_param_shape)
        if dilated_steps > 0:
            self.dilated_rollouts = ReinforcementStorage(num_options, dilation_queue_len * dilated_start, extracted_shape, current_shape, changepoint_shape, action_space, resp_len, option_param_shape, gamma_dilation = dilated_start)
            self.target_rollouts = ReinforcementStorage(num_options, dilation_queue_len * target_start, extracted_shape, current_shape, changepoint_shape, action_space, resp_len, option_param_shape)
        if trace_queue_len > 0:
            self.trace_rollouts = ReinforcementStorage(num_options, trace_queue_len, extracted_shape, current_shape, changepoint_shape, action_space, resp_len, option_param_shape)

    # def set_changepoint_queue(self, newlen): # TODO: currently reset queue as well
    #     self.changepoint_queue = torch.zeros(newlen, *self.changepoint_shape).detach()
    #     self.changepoint_action_queue = torch.zeros(newlen, 1).long().detach()
    #     self.changepoint_queue_len = newlen

    def set_parameters(self, num_steps):
        self.num_steps = num_steps
        if self.buffer_steps < 0: # the return buffer does not need parameters to be set
            lag_rollout = ReinforcementStorage(save_length=self.lag_num, source_rollout=self.base_rollouts)
            lag_rollout.copy_values(0,self.lag_num,self.base_rollouts,self.base_rollouts.buffer_filled - self.lag_num,self.base_rollouts.buffer_filled)
            self.base_rollouts.reset_length(num_steps + self.lag_num)
            self.base_rollouts.copy_values(0,self.lag_num,lag_rollout,0,self.lag_num)
            self.base_rollouts.buffer_filled = self.lag_num
            self.buffer_at = 0
        # if num_steps > self.changepoint_queue_len:
        #     self.set_changepoint_queue(num_steps)
        # print("returns", self.num_options, self.returns.shape, num_steps)

    def get_current(self):
        return self.base_rollouts.buffer_get_last(0, self.num_steps)

    def cuda(self):
        # self.states = self.states.cuda()
        self.iscuda =True
        self.base_rollouts.cuda()
        self.dilated_rollouts.cuda()
        self.trace_rollouts.cuda()

    def cpu(self):
        # self.states = self.states.cuda()
        self.iscuda = False
        self.base_rollouts.cpu()
        self.dilated_rollouts.cpu()
        self.trace_rollouts.cpu()

    def insert(self, reenter, extracted_state, current_state, action_probs, action, q_vals, value_preds, option_param, option_no, changepoint_state, epsilon, resp, done=False, option_agnostic = False):
        self.base_rollouts.insert(reenter, extracted_state, current_state, action_probs, action, q_vals, value_preds, option_param, option_no, changepoint_state, epsilon, resp, done=False, option_agnostic = False)
        self.buffer_at = self.base_rollouts.buffer_filled - self.lag_num

    def insert_trace(self, traces): # TODO: enforce trace diversity, enforce trace length
        reward_at = 0
        # print(self.trace_queue_len, self.rewards, torch.max(self.rewards))
        if self.trace_queue_len > 0:
            for oidx, option_reward in enumerate(self.rewards):
                removed_amount = 0
                while torch.max(option_reward) > 1:
                    reward_at = len(traces) - len(option_reward) + option_reward.argmax() + removed_amount
                    # print(traces[max(reward_at - self.trace_len,0):reward_at])
                    for i, (current_state, action) in enumerate(traces[max(reward_at - self.trace_len,0):reward_at]): # assuming no resets (choose a short trace)
                        self.trace_states[oidx, self.trace_at, i].copy_(current_state.squeeze())
                        self.trace_actions[oidx, self.trace_at, i].copy_(action.squeeze())
                    self.trace_at = (self.trace_at + 1) % self.trace_queue_len 
                    self.trace_filled += int(self.trace_filled < self.trace_queue_len)
                    removed_amount = reward_at
        return traces[reward_at+1:]

    def insert_dilation(self, swap): # must be run as often as insert
        if swap:
            if self.dilation_filled == self.dilation_queue_len:
                self.dilated_indexes.roll(1, axis=0)
            self.dilated_counter = self.dilated_start
            self.dilated_indexes[self.dilation_filled].copy_(torch.tensor(self.dilated_rollouts.buffer_filled)+1)
            self.dilation_filled += self.dilation_filled < self.dilation_queue_len
        if self.dilated_counter > 0:
            self.dilated_rollouts.insert_other(self.base_rollouts, self.base_rollouts.buffer_filled - 1)
            self.dilated_counter -= 1
            self.dilated_indexes -= 1 # assumes none of them goes under 0

    def insert_hindsight_target(self, change, target):
        if change:
            if self.change_filled == self.dilation_queue_len:
                self.dilated_change_targets.roll(1, axis=0)
                self.dilated_change_indexes.roll(1, axis=0)
                self.target_indexes.roll(1, axis=0)
            self.dilated_change_targets[self.change_filled].copy_(target.squeeze)
            self.dilated_change_indexes[self.change_filled].copy_(self.dilated_indexes[self.dilation_filled-1])
            self.change_filled += self.change_filled < self.dilation_queue_len
            self.target_counter = self.target_start // 2
            if self.target_counter == 0:
                for i in reversed(range(1, self.target_start // 2)):
                    self.target_rollouts.insert_other(self.base_rollouts, self.base_rollouts.buffer_filled - 1 - i)
            self.target_indexes[self.change_filled].copy(torch.tensor(self.target_rollouts.buffer_filled)+1)
        elif self.target_counter > 0:
            self.target_rollouts.insert_other(self.base_rollouts, self.base_rollouts.buffer_filled - 1)
            self.target_counter -= 1
            self.target_indexes -= 1 # assumes none of them goes under 0


    def insert_rewards(self, args, rewards):
        self.base_rollouts.insert_rewards(args, rewards)

    def buffer_get_last(self, i, n, changepoint=False):
        cp_states, actions, current_states, returns = self.base_rollouts.buffer_get_last(i,n)
        return cp_states, actions, current_states, returns
