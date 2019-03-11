import numpy as np
from file_management import get_edge
from collections import Counter
from ReinforcementLearning.models import pytorch_model
import torch

class Novelty_Wrapper():
    def __init__(self, args, reward_function, minmax=None):
        '''
        wraps a novelty reward over an existing reward function
        '''
        self.name = reward_function.name
        self.head, self.tail = get_edge(args.train_edge)
        self.reward_function = reward_function
        self.cuda = args.cuda
        self.traj_dim = reward_function.traj_dim #TODO: the dimension of the input trajectory is currently pre-set at 2, the dim of a location. Once we figure out dynamic setting, this can change
        self.novelty_decay = args.novelty_decay

    def compute_reward(self, states, actions):
        '''
        takes in states, actions in format: [num in batch (sequential), dim of state/action], there is one more state than action
        for state, action, nextstate
        returns rewards in format: [num in batch, 1]
        '''
        base_reward = self.reward_function.compute_reward(states, actions)
        return base_reward

    def get_trajectories(self, full_states): # head is first and tail is second
        return self.reward_function.get_trajectories(full_states)

class VisitationCountReward(Novelty_Wrapper):
    '''
    The most naieve novelty search. Gives reward based on how often a state has been seen before, according to the form

    '''
    def __init__(self, args, reward_function, minmax=None):
        super().__init__(args, reward_function)
        self.num_states = args.num_steps # to get unique states
        self.magnitude = args.visitation_magnitude # the reward for a never-visited before state
        self.lmbda = args.visitation_lambda # 
        self.seen_states = Counter()

    def hash(self, state):
        return tuple(int(v) for v in state)


    def compute_reward(self, states, actions):
        # TODO: decay rate so rewards come back later?
        reward = []
        for state in states[max(-self.num_states-1, -states.size(0)+1):-1]:
            seen_num = self.seen_states[self.hash(state)]
            self.seen_states[self.hash(state)] += 1
            reward = [self.lmbda/(seen_num + self.lmbda) * self.magnitude] + reward
        # print(reward,len(states), self.num_states)
        reward = pytorch_model.wrap([0 for i in range(len(states) - self.num_states - 2)] + reward)
        # print("reward", reward)
        base_reward = self.reward_function.compute_reward(states, actions)
        # print("base_reward", base_reward)
        # print(reward, base_reward)
        # print(list(self.seen_states.values()))
        # print(max(list(self.seen_states.values())))
        return reward + base_reward

class GaussianHashReward(VisitationCountReward):
    '''
    hashing based on distance, with each dimension separate. Basically the tile coding equivalent of above
    '''
    def __init__(self, args, reward_function, minmax):
        super().__init__(args, reward_function)
        self.minmax = (pytorch_model.wrap(minmax[0], cuda=args.cuda).detach(), pytorch_model.wrap(minmax[1], cuda=args.cuda).detach())
        minvs, maxvs = self.minmax
        self.order_vectors = []
        for minv, maxv in zip(minvs, maxvs):
            order_vector = []
            for i in range (args.novelty_hash_order):
                order_vector.append((i / (args.novelty_hash_order - 1)))
            self.order_vectors.append(pytorch_model.wrap(np.array(order_vector), cuda = args.cuda).detach())

    def normalize(self, x):
        return (x - self.minmax[0]) / (self.minmax[1] - self.minmax[0] + 1e-10)

    def hash(self, state):
        '''
        assuming state of the form [changepoint state dim]
        '''
        state = self.normalize(state)
        basis = []
        for order_vector, val in zip(self.order_vectors, state):
            basis.append(int(pytorch_model.unwrap(torch.exp(-(val - order_vector).pow(2)).argmax()))) # could use any monotonically decreasing function
        return tuple(basis)

class CorrelateHashReward(VisitationCountReward):
    '''
    hashes on the correlate position, which is essentially encouraging diversity of what we can control
    '''
    def __init__(self, args, reward_function, minmax):
        super().__init__(args, reward_function)
        self.seen_states = Counter()
        self.minmax = (pytorch_model.wrap(minmax[0][-self.traj_dim:], cuda=args.cuda).detach(), pytorch_model.wrap(minmax[1][-self.traj_dim:], cuda=args.cuda).detach())
        minvs, maxvs = self.minmax
        self.order_vectors = []
        for minv, maxv in zip(minvs, maxvs): # TODO: assumes there is only one other object
            order_vector = [] # TODO: assums that minmax is representative of the minimum and maximum possible 
            for i in range (args.novelty_hash_order):
                order_vector.append(i / (args.novelty_hash_order - 1))
            self.order_vectors.append(pytorch_model.wrap(np.array(order_vector), cuda = args.cuda).detach())

    def normalize(self, x):
        return (x - self.minmax[0]) / (self.minmax[1] - self.minmax[0] + 1e-10)

    def hash(self, state):
        '''
        assuming state of the form [changepoint state dim]
        '''
        state = self.normalize(state[-self.traj_dim:])
        basis = []
        for order_vector, val in zip(self.order_vectors, state):
            basis.append(int(pytorch_model.unwrap(torch.exp(-(val - order_vector).pow(2)).argmax()))) # could use any monotonically decreasing function
        return tuple(basis)


novelty_rewards = {"count": VisitationCountReward, "tile": GaussianHashReward, "correlate": CorrelateHashReward}