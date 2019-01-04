import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import sys, glob, copy, os, collections, time
from arguments import get_args
from learning_algorithms import *
from models import *

class pytorch_model():
    def __init__(self, combiner=None, loss=None, reducer=None, cuda=True):
        # should have customizable combiner and loss, but I dont.
        self.cuda=cuda
        self.reduce_size = 2 # someday this won't be hard coded either

    @staticmethod
    def wrap(data, dtype=torch.float, cuda=True):
        # print(Variable(torch.Tensor(data).cuda()))
        if cuda:
            return Variable(torch.tensor(data, dtype=dtype).cuda())
        else:
            return Variable(torch.tensor(data, dtype=dtype))

    @staticmethod
    def unwrap(data):
        return data.data.cpu().numpy()

    @staticmethod
    def concat(data, axis=0):
        return torch.cat(data, dim=axis)

class RolloutOptionStorage(object):
    def __init__(self, num_processes, obs_shape, action_space,
     extracted_shape, current_shape, buffer_steps, changepoint_queue_len, num_options):
        # TODO: storage does not currently support multiple processes, can be implemented
        self.num_processes = num_processes
        self.obs_shape = obs_shape
        self.action_space = action_space
        self.extracted_shape = extracted_shape
        self.current_shape = current_shape
        self.buffer_steps = buffer_steps
        self.changepoint_queue_len = changepoint_queue_len
        self.buffer_filled = 0
        self.buffer_at = 0
        self.return_at = 0
        self.changepoint_at = 0
        self.state_queue = torch.zeros(buffer_steps, *self.extracted_shape)
        self.current_state_queue = torch.zeros(buffer_steps, *self.current_shape)
        self.action_probs_queue = torch.zeros(num_options, buffer_steps, action_space)
        self.Qvals_queue = torch.zeros(num_options, buffer_steps, action_space)
        self.reward_queue = torch.zeros(num_options, buffer_steps, 1)
        self.return_queue = torch.zeros(num_options, buffer_steps, 1)
        self.values_queue = torch.zeros(num_options, buffer_steps, 1)
        self.action_queue = torch.zeros(buffer_steps, 1).long()
        self.option_queue = torch.zeros(buffer_steps, 1).long()
        self.changepoint_queue = torch.zeros(changepoint_queue_len, *self.extracted_shape)
        self.num_options = num_options
        self.iscuda = False
        self.set_parameters(1)
        self.last_step = 0

    def set_parameters(self, num_steps):
        self.last_step = num_steps - 1
        self.extracted_state = torch.zeros(num_steps + 1, *self.extracted_shape)
        self.current_state = torch.zeros(num_steps + 1, *self.current_shape)
        self.rewards = torch.zeros(self.num_options, num_steps, 1) # Pretend there are no other processes
        self.returns = torch.zeros(self.num_options, num_steps + 1, 1)
        self.action_probs = torch.zeros(self.num_options, num_steps + 1, self.action_space)
        self.Qvals = torch.zeros(self.num_options, num_steps + 1, self.action_space)
        self.value_preds = torch.zeros(self.num_options, num_steps + 1, 1)
        self.actions = torch.zeros(num_steps + 1, 1) # no other processes
        self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, *self.obs_shape) # no other processes

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
            self.changepoint_queue = self.changepoint_queue.cuda()
            self.current_state_queue = self.current_state_queue.cuda()
        self.current_state = self.current_state.cuda()
        self.extracted_state = self.extracted_state.cuda()
        self.action_probs = self.action_probs.cuda()
        self.Qvals = self.Qvals.cuda()
        self.rewards = self.rewards.cuda()
        self.value_preds = self.value_preds.cuda()
        self.returns = self.returns.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()

    def insert(self, step, extracted_state, current_state, action_probs, action, q_vals, value_preds, option_no): # got rid of masks... might be useful though
        if self.buffer_steps > 0 and self.last_step != step: # using buffer and not the first step, which is a duplicate of the last step
            self.buffer_filled += int(self.buffer_filled < self.buffer_steps)
            self.state_queue[self.buffer_at].copy_(extracted_state.squeeze())
            self.current_state_queue[self.buffer_at].copy_(current_state.squeeze())
            self.option_queue[self.buffer_at].copy_(pytorch_model.wrap(option_no, cuda=self.iscuda))
            self.action_queue[self.buffer_at].copy_(action.squeeze())
            for oidx in range(self.num_options):
                self.values_queue[oidx, self.buffer_at].copy_(value_preds[oidx].squeeze())
                self.Qvals_queue[oidx, self.buffer_at].copy_(q_vals[oidx].squeeze())
                self.action_probs_queue[oidx, self.buffer_at].copy_(action_probs[oidx].squeeze())
            if self.buffer_at == self.buffer_steps - 1:
                self.buffer_at = 0
            else:
                self.buffer_at += 1
        if self.changepoint_queue_len > 0 and self.last_step != step:
            if self.changepoint_at == self.changepoint_queue_len - 1:
                self.changepoint_at = 0
            else:
                self.changepoint_at += 1
            self.changepoint_queue[self.changepoint_at].copy_(extracted_state.squeeze())
        self.extracted_state[step].copy_(extracted_state.squeeze())
        self.actions[step].copy_(action.squeeze())
        self.current_state[step].copy_(current_state.squeeze())
        for oidx in range(self.num_options):
            self.value_preds[oidx, step].copy_(value_preds[oidx].squeeze())
            self.Qvals[oidx, step].copy_(q_vals[oidx].squeeze())
            self.action_probs[oidx, step].copy_(action_probs[oidx].squeeze())

    def insert_rewards(self, rewards):
        if self.buffer_steps > 0:
            for oidx in range(rewards.size(0)):
                for i, reward in enumerate(rewards[oidx]):
                    self.reward_queue[oidx, (self.buffer_at + i - len(rewards) - 2) % self.buffer_steps].copy_(reward)
        self.rewards = rewards

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
            elif return_format == 2: # compute next set of returns from buffer
                self.returns[idx, -1] = 0
                for step in reversed(range(self.rewards.size(1))):
                    self.returns[idx, step] = self.returns[idx, step + 1] * gamma + self.rewards[idx, step] # compute most recent returns
                for i, ret in enumerate(self.returns):
                    # update the last ten returns
                    if 11-i > 0:
                        for j in range(i, 11-i): #1, 2, 3, 4 ... 2, 3, 4 ... 3, 4, 
                            # print(self.returns[0], self.return_queue[-6+j])
                            self.return_queue[idx, (-11+j+self.return_at) % self.buffer_steps] += torch.tensor(np.power(gamma, 10-j)).cuda() * self.returns[idx, i]
                for ret in self.returns:
                    self.return_queue[idx, self.return_at].copy_(ret)
                    # print(ret, self.return_queue[self.return_at])
                    self.return_at = (self.return_at + 1) % self.buffer_steps
                # print("return_queue", self.rewards)
            elif return_format == 3: # segmented returns
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

    def compute_full_returns(self, next_value, gamma):
        self.returns = torch.zeros(self.num_options, self.rewards.size(1) + 1, 1).cuda()
        for idx in range(self.num_options):
            self.returns[idx, -1] = 0 #next_value
            for step in reversed(range(self.rewards.size(1))):
                self.returns[idx, step] = self.returns[idx, step + 1] * gamma + self.rewards[idx, step]

class ChainMDP():
    def __init__(self, num_states):
        self.minmax = (0,num_states)
        self.num_states = num_states
        self.current_state = 0

    def step(self, action):
        if action == 0 and self.current_state > 0:
            self.current_state -= 1
        elif action == 1:
            pass
        elif action == 2 and self.current_state < self.num_states-1:
            self.current_state += 1
        done = self.current_state == self.num_states - 1
        if done:
            self.current_state = 0
        return self.current_state, done

    def getState(self):
        return self.current_state

class ProxyEnvironment():
    def __init__(self, args, proxy_chain, reward_fns, state_get):
        '''
        an environment with (sub)states that are a subspace of the true states, actions that are options with a shared state space,
        and rewards which are based on the substates
        '''
        self.proxy_chain = proxy_chain
        self.reward_fns = reward_fns
        self.stateExtractor = state_get
        self.args = args

        self.num_hist = args.num_stack
        self.state_size = self.stateExtractor.state_size
        self.action_size = self.stateExtractor.action_size
        self.reset_history()
        self.extracted_state = self.stateExtractor.get_state(proxy_chain[0].getState())
        self.insert_extracted()

    def set_models(self, models):
        self.models = models

    def reset_history(self):
        self.current_state = pytorch_model.wrap(np.zeros((self.num_hist * int(np.prod(self.state_size[0])), )), cuda = self.args.cuda).unsqueeze(0)
        # TODO: add multi-process code someday

    def insert_extracted(self):
        '''
        self.current_state has history, and is of shape: [batch/processor number, hist len * state size]
        '''
        shape_dim0 = self.num_hist # make sure this is 1 if no history is to be used
        state_size = int(np.prod(self.state_size))
        if self.num_hist > 1:
            self.current_state[:, :shape_dim0*state_size-1] = self.current_state[:, -shape_dim0*state_size+1:]
        self.current_state[:, shape_dim0*state_size-1:] = self.extracted_state
        return self.current_state

    def getState(self):
        return self.extracted_state

    def getHistState(self):
        return self.current_state

    def step(self, action, model=False):
        '''
        steps the true environment. The last environment in the proxy chain is the true environment,
        and has a different step function.
        model determines if action is a model 
        '''
        if model:
            action, values, action_probs, Q_vals = self.models.determine_action(self.current_state, index=action)
        if len(self.proxy_chain) > 1:
            state, done, raw_state = self.proxy_chain[-1].step(action, model=True)
        else:
            raw_state, done = self.proxy_chain[-1].step(action)
        if done:
            self.reset_history()
        self.raw_state = raw_state
        # TODO: implement multiprocessing support
        self.extracted_state = pytorch_model.wrap(self.stateExtractor.get_state(raw_state), cuda=self.args.cuda).unsqueeze(0)
        self.insert_extracted()
        return self.extracted_state, done, self.raw_state

    def computeReward(self, rollout):
        # probably doesn't have to be in here
        if rollout.changepoint_queue_len > 0 and rollout.use_queue:
            states = torch.cat([rollout.changepoint_queue[rollout.changepoint_at:], rollout.changepoint_queue[:rollout.changepoint_at]], dim=0) # multiple policies training
            actions = torch.cat([rollout.changepoint_action_queue[rollout.changepoint_at:], rollout.changepoint_action_queue[:rollout.changepoint_at]], dim=0)
        else:
            states = rollout.extracted_state
            actions = rollout.actions
        rewards = []
        for reward_fn in self.reward_fns:
            rewards.append(reward_fn.compute_reward(states,actions))
        return torch.stack(rewards, dim=0)


class OptionChain():
    def __init__(self, true_environment):
        # self.graph = graph
        self.true_environment = true_environment

    def initialize(self, args):
        '''
        TODO: make this generalizable to other cases.
        '''
        return [self.true_environment]

class RewardRight():
    def compute_reward(self, states, actions):
        '''

        TODO: make support multiple processes
        possibly make this not iterative?
        '''
        rewards = []
        for state, action, nextstate in zip(states, actions, states[1:]):
            # print(state)
            if state - nextstate == -1:
                rewards.append(0)
            else:
                rewards.append(-1)
        return pytorch_model.wrap(rewards, cuda=True)

def sample_actions( probs, deterministic):
    if deterministic is False:
        cat = torch.distributions.categorical.Categorical(probs.squeeze())
        action = cat.sample()
        action = action.unsqueeze(-1).unsqueeze(-1)
    else:
        action = probs.max(1)[1]
    return action

class MultiOption():
    def __init__(self, num_options, option_class): 
        self.option_index = 0
        self.num_options = num_options
        self.option_class = option_class

    def initialize(self, args, num_options, state_class):
        self.models = []
        for i in range(num_options):
            model = self.option_class(args, state_class.flat_state_size * args.num_stack, state_class.action_size, name = args.unique_id + str(i), minmax = state_class.get_minmax())
            if args.cuda:
                model = model.cuda()
            self.models.append(model) # make this an argument controlled parameter
        self.option_index = 0

    def names(self):
        return [model.name for model in self.models]

    def determine_action(self, state, index=-1):
        if index == -1:
            index = self.option_index
        return self.models[index](state)

    def currentName(self):
        return self.models[self.option_index].name

class EpsilonGreedyQ():

    def initialize(self, args, num_outputs):
        self.epsilon = args.greedy_epsilon
        self.num_outputs = num_outputs

    def take_action(self, probs, q_vals):
        action = sample_actions(F.softmax(q_vals, dim=1), deterministic =True)
        if np.random.rand() < .1:
            action = pytorch_model.wrap(np.random.randint(self.num_outputs, size = q_vals.shape[0]), cuda = True)
        return action

class EpsilonGreedyProbs():

    def initialize(self, args, num_outputs):
        self.epsilon = args.greedy_epsilon
        self.num_outputs = num_outputs

    def take_action(self, probs, q_vals):
        action = sample_actions(probs, deterministic =True)
        if np.random.rand() < .1:
            action = pytorch_model.wrap(np.random.randint(self.num_outputs, size = probs.shape[0]), cuda = True)
        return action

class GetRaw():
    def __init__(self, raw_state_shape, raw_action_num, raw_minmax):
        self.state_size = raw_state_shape
        self.flat_state_size = np.prod(raw_state_shape)
        self.action_size = raw_action_num
        self.minmax = raw_minmax

    def get_state(self, state):
        return state

    def get_minmax(self):
        return self.minmax


def unwrap_or_none(val):
    if val is not None:
        return pytorch_model.unwrap(val)
    else:
        return -1.0

def trainRL(args, true_environment, train_models, learning_algorithm, 
            option_chain, reward_classes, state_class, behavior_policy):
    print("#######")
    print("Training Options")
    print("#######")
    # if option_chain is not None: #TODO: implement this
    save_path = os.path.join(args.save_dir, args.unique_id)
    proxy_chain = option_chain.initialize(args)
    proxy_environment = ProxyEnvironment(args, proxy_chain, reward_classes, state_class)
    behavior_policy.initialize(args, state_class.action_size)
    train_models.initialize(args, len(reward_classes), state_class)
    proxy_environment.set_models(train_models)
    learning_algorithm.initialize(args, train_models)
    state = pytorch_model.wrap(proxy_environment.getState(), cuda = args.cuda)
    hist_state = pytorch_model.wrap(proxy_environment.getHistState(), cuda = args.cuda)
    rollouts = RolloutOptionStorage(args.num_processes, state_class.state_size, state_class.action_size, state.shape, hist_state.shape, args.buffer_steps, args.changepoint_queue_len, len(train_models.models))
    option_actions = {option.name: collections.Counter() for option in train_models.models}
    total_duration = 0
    total_elapsed = 0
    start = time.time()
    fcnt = 0
    final_rewards = list()
    option_counter = collections.Counter()
    option_value = collections.Counter()
    for j in range(args.num_iters):
        rollouts.set_parameters(learning_algorithm.current_duration)
        raw_actions = []
        rollouts.cuda()
        for step in range(learning_algorithm.current_duration):
            fcnt += 1
            current_state = proxy_environment.getHistState()
            values, dist_entropy, action_probs, Q_vals = train_models.determine_action(current_state)
            action = behavior_policy.take_action(action_probs, Q_vals)
            rollouts.insert(step, state, current_state, action_probs, action, Q_vals, values, train_models.option_index)
            state, done, raw_state = proxy_environment.step(action, model = False)#, render=len(args.record_rollouts) != 0, save_path=args.record_rollouts, itr=fcnt)
            learning_algorithm.interUpdateModel(step)

            #### logging
            option_actions[train_models.currentName()][int(pytorch_model.unwrap(action.squeeze()))] += 1
            #### logging

            if done:
                print("reached end")
                break
        current_state = proxy_environment.getHistState()
        values, dist_entropy, action_probs, Q_vals = train_models.determine_action(current_state)
        action = behavior_policy.take_action(action_probs, Q_vals)
        rollouts.insert(step + 1, state, current_state, action_probs, action, Q_vals, values, train_models.option_index) # inserting the last state and unused action
        total_duration += step + 1
        rewards = proxy_environment.computeReward(rollouts)
        rollouts.insert_rewards(rewards)
        rollouts.compute_returns(args, values)
        name = train_models.currentName()

        #### logging
        reward_total = rollouts.rewards.sum(dim=1)[train_models.option_index]
        final_rewards.append(reward_total)
        option_counter[name] += step + 1
        option_value[name] += reward_total.data  
        #### logging

        value_loss, action_loss, dist_entropy, output_entropy, entropy_loss, action_log_probs = learning_algorithm.step(args, train_models, rollouts) 
        learning_algorithm.updateModel()
        if j % args.save_interval == 0 and args.save_dir != "" and args.train: # no point in saving if not training
            train_models.save(args) # TODO: implement save_options

        #### logging
        if j % args.log_interval == 0:
            for name in train_models.names():
                if option_counter[name] > 0:
                    print(name, option_value[name] / option_counter[name], [option_actions[name][i]/option_counter[name] for i in range(len(option_actions[name]))])
                if j % (args.log_interval * 20) == 0:
                    option_value[name] = 0
                    option_counter[name] = 0
                    for i in range(len(option_actions[name])):
                        option_actions[name][i] = 0
            end = time.time()
            final_rewards = np.array(final_rewards)
            el, vl, al = unwrap_or_none(entropy_loss), unwrap_or_none(value_loss), unwrap_or_none(action_loss)
            total_elapsed += total_duration
            log_stats = "Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {}, value loss {}, policy loss {}".format(j, total_duration,
                       int(total_elapsed / (end - start)),
                       final_rewards.mean(),
                       np.median(final_rewards),
                       final_rewards.min(),
                       final_rewards.max(), el,
                       vl, al)
            print(log_stats)
            final_rewards = list()
            total_duration = 0
        #### logging

if __name__ == "__main__":
    args = get_args()
    true_environment = ChainMDP(100)
    # train_models = MultiOption(1, BasicModel)
    train_models = MultiOption(1, TabularQ)
    # learning_algorithm = DQN_optimizer()
    # learning_algorithm = DDPG_optimizer()
    option_chain = OptionChain(true_environment)
    reward_classes = [RewardRight()]
    state_class = GetRaw((1,), 3, true_environment.minmax)
    # behavior_policy = EpsilonGreedyQ()
    behavior_policy = EpsilonGreedyProbs()
    trainRL(args, true_environment, train_models, learning_algorithm, 
            option_chain, reward_classes, state_class, behavior_policy=behavior_policy)
