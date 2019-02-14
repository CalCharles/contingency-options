import torch
from ReinforcementLearning.models import pytorch_model
import numpy as np
from RewardFunctions.changepointReward import ChangepointReward
from file_management import get_edge


class BounceReward(ChangepointReward):
    def __init__(self, vel, args): 
        self.anybounce = False
        self.desired_vels = [torch.tensor([-2.,-1.]).cuda(), torch.tensor([-1.,-1.]).cuda(), torch.tensor([-1.,1.]).cuda(), torch.tensor([-2.,1.]).cuda()]
        if vel == -1:
            self.anybounce = True
        self.desired_vel = self.desired_vels[0]
        if vel == 0:
            self.desired_vel = self.desired_vels[0]
        elif vel == 1:
            self.desired_vel = self.desired_vels[1]
        elif vel == 2:
            self.desired_vel = self.desired_vels[2]
        elif vel == 3:
            self.desired_vel = self.desired_vels[3]
        self.traj_dim = 2 # SET THIS
        self.head, self.tail = get_edge(args.train_edge)


    def compute_reward(self, states, actions):
        '''
        states must have at least two in the stack: to keep size of rewards at num_states - 1
        assumes ball is the last state
        assuming input shape: [state_size = num_stack*traj_dim]
        '''
        rewards = []
        # print(states.shape)
        for last_state, state, action, nextstate in zip(states, states[1:], actions, states[2:]):
            last_state = last_state.squeeze()
            state = state.squeeze()
            nextstate = nextstate.squeeze()
            state_first = last_state[:2]
            state_second = state[:2]
            proximity = state[:2] - state[-2:]
            state_third = nextstate[:2]
            # print(state_second.shape, state.shape, state_first.shape)
            v1 = state_second - state_first
            v2 = state_third - state_second
            # print(state_first, state_second, state_third)
            rewarded = False
            if v1[0] > 0 and state_second[0] > 65: # was moving down, below the blocks
                if torch.norm(v2 - self.desired_vel) == 0:
                    rewards.append(10)
                    rewarded = True
                elif self.anybounce:
                    for v in self.desired_vels:
                        if torch.norm(v2 - v) == 0:
                            # print ("REWARD", v1, v2)
                            rewards.append(10)
                            rewarded = True
            if not rewarded:
                rewards.append(-abs(proximity[0] / (proximity[1] + .05) * .05))
        return pytorch_model.wrap(rewards, cuda=True)

class Xreward(ChangepointReward):
    def __init__(self, args): 
        self.traj_dim = 2 # SET THIS
        self.head, self.tail = get_edge(args.train_edge)


    def compute_reward(self, states, actions):
        '''
        states must have at least two in the stack: to keep size of rewards at num_states - 1
        assumes ball is the last state
        assuming input shape: [state_size = num_stack*traj_dim]
        '''
        rewards = []
        # print(states.shape)
        for last_state, state, action, nextstate in zip(states, states[1:], actions, states[2:]):
            base = state.squeeze()[:2]
            corr = state.squeeze()[-2:]
            # print(base, corr)
            state = base-corr
            # print(state, -abs(int(state[1])))
            rewards.append(-abs(int(state[1])))
        return pytorch_model.wrap(rewards, cuda=True)



class RewardRight(ChangepointReward):
    def compute_reward(self, states, actions):
        '''

        TODO: make support multiple processes
        possibly make this not iterative?
        '''
        rewards = []
        for state, action, nextstate in zip(states, actions, states[1:]):
            # print(state)
            if state - nextstate == -1:
                rewards.append(2)
            else:
                rewards.append(-1)
        return pytorch_model.wrap(rewards, cuda=True)

class RewardLeft(ChangepointReward):
    def compute_reward(self, states, actions):
        '''

        TODO: make support multiple processes
        possibly make this not iterative?
        '''
        rewards = []
        for state, action, nextstate in zip(states, actions, states[1:]):
            # print(state)
            if state - nextstate == 1:
                rewards.append(2)
            else:
                rewards.append(-1)
        return pytorch_model.wrap(rewards, cuda=True)


class RewardCenter(ChangepointReward):
    def compute_reward(self, states, actions):
        '''

        TODO: make support multiple processes
        possibly make this not iterative?
        '''
        rewards = []
        for state, action, nextstate in zip(states, actions, states[1:]):
            # print(state)
            if state - nextstate == 0:
                rewards.append(2)
            else:
                rewards.append(-1)
        return pytorch_model.wrap(rewards, cuda=True)
