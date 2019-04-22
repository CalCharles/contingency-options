import numpy as np
from file_management import get_edge
from collections import Counter
from Models.models import pytorch_model
import torch

def construct_tile_order(minmax, normalize, order):
    minvs, maxvs = minmax
    order_vectors = []
    for minv, maxv in zip(minvs, maxvs):
        order_vector = []
        numv = min(order, int(pytorch_model.unwrap(torch.ceil(maxv - minv) + 1))) # TODO: assumes integer differences between states, fix?
        for i in range (numv): 
            if not normalize:
                order_vector.append((minv + i * (maxv - minv) / (max(numv - 1, 1))))
            else:
                order_vector.append((i / max(numv - 1, 1)))
        order_vectors.append(pytorch_model.wrap(np.array(order_vector)).detach())
    for vec in order_vectors:
        vec.requires_grad = False   
    return order_vectors

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

    def compute_reward(self, states, actions, resps):
        '''
        takes in states, actions in format: [num in batch (sequential), dim of state/action], there is one more state than action
        for state, action, nextstate
        returns rewards in format: [num in batch, 1]
        '''
        base_reward = self.reward_function.compute_reward(states, actions, resps)
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


    def compute_reward(self, states, actions, resps):
        # TODO: decay rate so rewards come back later?
        reward = []
        for state in states[max(-self.num_states-1, -states.size(0)+1):-1]:
            hsh = self.hash(state)
            seen_num = self.seen_states[hsh]
            self.seen_states[hsh] += 1
            reward = [self.lmbda/(seen_num + self.lmbda) * self.magnitude] + reward
        # print(reward,len(states), self.num_states)
        reward = pytorch_model.wrap([0 for i in range(len(states) - self.num_states - 2)] + reward)
        # print("reward", reward)
        base_reward = self.reward_function.compute_reward(states, actions, resps)
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
        self.order_vectors = construct_tile_order(self.minmax, True, args.novelty_hash_order)

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
        self.order_vectors = construct_tile_order(self.minmax, True, args.novelty_hash_order)

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

class BaseRewardNovelty(CorrelateHashReward):
    '''
    hashes on the correlate position, which is essentially encouraging diversity of reward locations
    '''
    def __init__(self, args, reward_function, minmax):
        super().__init__(args, reward_function, minmax)

    def compute_reward(self, states, actions, resps):
        # TODO: decay rate so rewards come back later?
        base_reward = self.reward_function.compute_reward(states, actions, resps)
        for i in range(len(base_reward) - min(self.num_states, states.size(0)-2),len(base_reward)):
            reward_at = base_reward[i]
            state = states[i]
            if reward_at > 0:
                hsh = self.hash(state)
                seen_num = self.seen_states[hsh]
                self.seen_states[hsh] += 1
                base_reward[i] = self.lmbda/(seen_num + self.lmbda) * self.magnitude + self.magnitude * base_reward[i]
        return base_reward


novelty_rewards = {"count": VisitationCountReward, "tile": GaussianHashReward, "correlate": CorrelateHashReward, "reward": BaseRewardNovelty}