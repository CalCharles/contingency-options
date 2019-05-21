import torch
import time
from Models.models import pytorch_model
import numpy as np
from RewardFunctions.changepointReward import ChangepointReward
from file_management import get_edge
from Environments.state_definition import GetState


class BounceReward(ChangepointReward):
    def __init__(self, vel, args):
        super().__init__(None, args)
        self.name = "Paddle->Ball"
        self.head, self.tail = "Ball", "Paddle"

        self.anybounce = False
        self.desired_vels = [pytorch_model.wrap([-2.,-1.], cuda=args.cuda), pytorch_model.wrap([-1.,-1.], cuda=args.cuda), pytorch_model.wrap([-1.,1.], cuda=args.cuda), pytorch_model.wrap([-2.,1.], cuda=args.cuda)]
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
        self.form = args.reward_form


    def compute_reward(self, states, actions, resps):
        '''
        states must have at least two in the stack: to keep size of rewards at num_states - 1
        assumes ball is the last state
        assuming input shape: [state_size = num_stack*traj_dim]
        '''
        rewards = []
        # print(states.shape)
        # start = time.time()
        for last_state, state, action, nextstate in zip(states, states[1:], actions, states[2:]):
            last_state = last_state.squeeze()
            state = state.squeeze()
            nextstate = nextstate.squeeze()
            state_first = last_state[:2]
            state_second = state[:2]
            proximity = state[:2] - state[-2:]
            state_third = nextstate[:2]
            # s1 = time.time()
            # print("separate ", s1 - start)
            # print(state_second.shape, state.shape, state_first.shape)
            v1 = state_second - state_first
            v2 = state_third - state_second
            # print(state_first, state_second, state_third)
            rewarded = False
            if v1[0] > 0 and state_second[0] > 65: # was moving down, below the blocks
                if torch.norm(v2 - self.desired_vel) == 0:
                    rewards.append(1)
                    rewarded = True
                else:
                    for v in self.desired_vels:
                        if torch.norm(v2 - v) == 0:
                            # print ("REWARD", v1, v2)
                            if self.anybounce:
                                rewards.append(1)
                            else:
                                rewards.append(0.25)
                            rewarded = True
            # s2 = time.time()
            # print("rew ", s1 - s2)
            if not rewarded:
                if self.form == 'dense':
                    # rewards.append(-abs(proximity[1] / (proximity[0] + .1) * .1))
                    rewards.append(-abs(proximity[0] + proximity[1]) * 0.001)
                if self.form.find('xdense') != -1:
                    if proximity[0] == 3 and self.form.find('neg') != -1:
                        rewards.append(-1)
                    # rewards.append(-abs(proximity[1] / (proximity[0] + .1) * .1))
                    else:
                        rewards.append(-abs(proximity[1]) * 0.001)
                else:
                    # print(state, proximity[0])
                    if proximity[0] > 3 and self.form.find('neg') != -1:
                        rewards.append(-1)
                    else:
                        rewards.append(0)
            # print("prewrap ", time.time() - s2)
        return pytorch_model.wrap(rewards, cuda=self.cuda)

class Xreward(ChangepointReward):
    def __init__(self, args): 
        super().__init__(None, args)
        self.traj_dim = 2 # SET THIS
        self.head, self.tail = get_edge(args.train_edge)
        self.name = "x"

    def compute_reward(self, states, actions, resps):
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

class BlockReward(ChangepointReward):
    def __init__(self, args): 
        super().__init__(None, args)
        self.form = args.reward_form
        self.state_class = GetState(target="Block", state_forms=[("Block", "multifull"), ("Ball", "bounds")]) # should be a block state class
        self.parameters = np.array([0,0])
        self.max_dist = np.linalg.norm([60, 20])
        self.cuda = args.cuda
        self.parameter_minmax = [np.array([0,0]), np.array([84,84])]

    def compute_reward(self, states, actions, resps):
        rewards = torch.zeros(len(states))
        change_indexes, ats, st = self.state_class.determine_delta_target(pytorch_model.unwrap(states))
        if len(change_indexes) > 0:
            dists = np.linalg.norm(self.parameters - st[0])
            rewards[change_indexes[0]] = (self.max_dist - dists) / self.max_dist
        rewards[states[:, -2] == 79] = -1.0
        if self.cuda:
            rewards = rewards.cuda()
        return rewards

    def determineChanged(self, states, actions, resps):
        change_indexes, ats, states = self.state_class.determine_delta_target(pytorch_model.unwrap(states))
        change = len(change_indexes) > 0
        if change:
            return change, states[0]
        return change, None

    def get_possible_parameters(self, state):
        last_shape = self.state_class.shapes[(self.state_class.names[0], self.state_class.fnames[0])][0]
        state = state[:last_shape]
        # print(state, last_shape, state.shape, self.state_class.shapes[(self.state_class.names[-1], self.state_class.fnames[-1])])
        state = state.view(-1,3)
        idxes = state[:,2].nonzero()[:,0].squeeze()
        # print(idxes, state[idxes,:2])
        return state[idxes,:2]

    def get_trajectories(self, full_states):
        states = []
        resps = []
        for state in full_states:
            state, resp = self.state_class.get_state(state)
            states.append(state)
            resps.append(resp)
        return pytorch_model.wrap(np.stack(states), cuda=self.cuda)

class RewardRight(ChangepointReward):
    def compute_reward(self, states, actions, resps):
        '''

        TODO: make support multiple processes
        possibly make this not iterative?
        '''
        rewards = []
        for state, action, nextstate in zip(states, actions, states[1:]):
            # print(state, state - nextstate == -1)
            if state - nextstate == -1:
                rewards.append(1)
            else:
                rewards.append(0)
        return pytorch_model.wrap(rewards, cuda=True)

class RewardLeft(ChangepointReward):
    def compute_reward(self, states, actions, resps):
        '''

        TODO: make support multiple processes
        possibly make this not iterative?
        '''
        rewards = []
        for state, action, nextstate in zip(states, actions, states[1:]):
            # print(state)
            if state - nextstate == 1:
                rewards.append(1)
            else:
                rewards.append(0)
        return pytorch_model.wrap(rewards, cuda=True)


class RewardCenter(ChangepointReward):
    def compute_reward(self, states, actions, resps):
        '''

        TODO: make support multiple processes
        possibly make this not iterative?
        '''
        rewards = []
        for state, action, nextstate in zip(states, actions, states[1:]):
            # print(state)
            if state - nextstate == 0:
                rewards.append(1)
            else:
                rewards.append(0)
        return pytorch_model.wrap(rewards, cuda=True)

class RewardCorner(ChangepointReward):
    def compute_reward(self, states, actions, resps):
        '''

        TODO: make support multiple processes
        possibly make this not iterative?
        '''
        rewards = []
        for state, action, nextstate in zip(states, actions, states[1:]):
            # print(state)
            if state - nextstate == 1:
                rewards.append(1)
            else:
                rewards.append(0)
        return pytorch_model.wrap(rewards, cuda=True)

class RewardTarget(ChangepointReward):
    def __init__(self, model, args, target):
        super().__init__(model, args)
        self.target = target

    def compute_reward(self, states, actions, resps):
        '''

        TODO: make support multiple processes
        possibly make this not iterative?
        '''
        rewards = []
        for state, action, nextstate in zip(states, actions, states[1:]):
            # print(state)
            if np.linalg.norm(state - self.target) == 0:
                rewards.append(1)
            else:
                rewards.append(-0.01)
        return pytorch_model.wrap(rewards, cuda=True)
