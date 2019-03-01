import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import imageio as imio
from ReinforcementLearning.models import pytorch_model



class RawEnvironment():
    def __init__(self):
        self.num_actions = None # this must be defined
        self.itr = 0 # this is used for saving, and is set externally
        self.save_path = "" # save dir also is set externally

    def step(self, action):
        '''
        self.save_path is the path to which to save files, and self.itr is the iteration number to be used for saving.
        The format of saving is: folders contain the raw state, names are numbers, contain 2000 raw states each
        obj_dumps contains the factored state
        empty string for save_path means no saving state
        Takes in an action and returns:
            next raw_state (image or observation)
            next factor_state (dictionary of name of object to tuple of object bounding box and object property)
            done flag: if an episode ends, done is True
        '''
        pass

    def getState(self):
        '''
        Takes in an action and returns:
            current raw_state (dictionary of name of object to raw state)
            current factor_state (dictionary of name of object to tuple of object bounding box and object property)
        '''
        pass

    def set_save(self, itr, save_dir, recycle):
        self.save_path=save_dir
        self.itr = itr
        self.recycle = recycle
        try:
            os.makedirs(save_dir)
        except OSError:
            pass


class ChainMDP(RawEnvironment):
    def __init__(self, num_states):
        super(ChainMDP, self).__init__()
        self.minmax = (0,num_states)
        self.num_states = num_states
        self.initial_state = np.array([self.num_states//2])
        self.current_state = self.initial_state

    def step(self, action):
        if action == 0 and self.current_state[0] > 0:
            v = self.current_state[0] - 1
            self.current_state = np.array([v])
        elif action == 1:
            pass
        elif action == 2 and self.current_state[0] < self.num_states-1:
            v = self.current_state[0] + 1
            self.current_state = np.array([v])
        done = self.current_state[0] == self.num_states - 1
        if len(self.save_path) != 0:
            state_path = os.path.join(self.save_path, str(self.itr//2000))
            try:
                os.makedirs(state_path)
            except OSError:
                pass
            # imio.imsave(os.path.join(state_path, "state" + str(self.itr % 2000) + ".png"), self.current_state)
            # print(self.save_path, state_path)
            if self.itr != 0:
                object_dumps = open(self.save_path + "/object_dumps.txt", 'a')
            else:
                object_dumps = open(self.save_path + "/object_dumps.txt", 'w') # create file if it does not exist
            # print("writing", self.save_path + "/object_dumps.txt")
            object_dumps.write("chain:"+str(self.current_state[0]) + "\t\n")
            object_dumps.close()
        self.itr += 1
        if self.itr % 20 == 0:
            self.current_state =self.initial_state

        # if done:
        #     self.current_state[0] = 0
        return self.current_state, {"chain": (self.current_state, 1)}, done

    def getState(self):
        return self.current_state, {"chain": (self.current_state, 1)}


class ProxyEnvironment():
    def __init__(self):
        '''
        create a dummy placeholder
        '''

    def initialize(self, args, proxy_chain, reward_fns, state_get, behavior_policy):
        '''
        an environment with (sub)states that are a subspace of the true states, actions that are options with a shared state space,
        and rewards which are based on the substates
        proxy_chain is the remainder of the proxy chain, after this environment
        reward_fns are the reward functions that specify options at this edge represented by this proxy environment
        state_get is the state extraction class for this edge
        In order to create a classic RL system, just use a single element list containing the true environment as the proxy chain
        '''
        self.proxy_chain = proxy_chain
        self.reward_fns = reward_fns
        self.stateExtractor = state_get
        self.iscuda = args.cuda
        self.name = args.unique_id # name should be of form: head_tail

        self.num_hist = args.num_stack
        self.state_size = self.stateExtractor.shape
        self.action_size = self.stateExtractor.action_num
        self.behavior_policy = behavior_policy
        self.reset_history()
        self.extracted_state = torch.Tensor(self.stateExtractor.get_state(proxy_chain[0].getState())).cuda()
        self.insert_extracted()

    def set_models(self, models):
        self.models = models

    def set_proxy_chain(self, proxy_chain):
        self.proxy_chain = proxy_chain

    def set_test(self):
        self.behavior_policy.set_test()


    def reset_history(self):
        self.current_state = pytorch_model.wrap(np.zeros((self.num_hist * int(np.prod(self.state_size)), )), cuda = self.iscuda)
        # TODO: add multi-process code someday

    def insert_extracted(self):
        '''
        self.current_state has history, and is of shape: [hist len * state size] TODO: [batch/processor number, hist len * state size]
        '''
        shape_dim0 = self.num_hist # make sure this is 1 if no history is to be used
        state_size = int(np.prod(self.state_size))
        if self.num_hist > 1:
            self.current_state[:(shape_dim0-1)*state_size] = self.current_state[-(shape_dim0-1)*state_size:]
        self.current_state[-state_size:] = self.extracted_state # unsqueeze 0 is for dummy multi-process code
        return self.current_state

    def getState(self):
        return self.extracted_state

    def getHistState(self):
        return self.current_state

    def step(self, action, model=False, action_list=[]):
        '''
        steps the true environment. The last environment in the proxy chain is the true environment,
        and has a different step function.
        raw_state is the tuple (raw_state, factor_state)
        model determines if action is a model 
        extracted state is the proxy extracted state, raw state is the full raw state (raw_state, factored state),
        done defines if a trajectory ends, action_list is all the actions taken by all the options in the chain
        '''
        if model:
            values, dist_entropy, probs, Q_vals = self.models.determine_action(self.current_state)
            action_probs, Q_vs = models.get_action(probs, Q_vals, index = action)
            action = self.behavior_policy.take_action(probs, Q_vals)
            # if issubclass(self.models.currentModel(), DopamineModel): 
            #     reward = self.computeReward(rollout, 1)
            #     action = self.models.currentModel().forward(self.current_state, reward[self.models.option_index])
        if len(self.proxy_chain) > 1:
            state, base_state, done, action_list = self.proxy_chain[-1].step(action, model=True, action_list = [action] + action_list)
        else:
            raw_state, factored_state, done = self.proxy_chain[-1].step(action)
            action_list = [action] + action_list

        if done:
            self.reset_history()
        self.raw_state = (raw_state, factored_state)
        # TODO: implement multiprocessing support
        self.extracted_state = pytorch_model.wrap(self.stateExtractor.get_state(self.raw_state), cuda=self.iscuda).unsqueeze(0)
        self.insert_extracted()
        return self.extracted_state, self.raw_state, done, action_list

    def step_dope(self, action, rollout, model=False, action_list=[]):
        '''
        steps the true environment, using dopamine models. The last environment in the proxy chain is the true environment,
        and has a different step function. 
        raw_state is the tuple (raw_state, factor_state)
        model determines if action is a model 
        extracted state is the proxy extracted state, raw state is the full raw state (raw_state, factored state),
        done defines if a trajectory ends, action_list is all the actions taken by all the options in the chain
        '''
        if model:
            reward = self.computeReward(rollout, 1)
            action = self.models.currentModel().forward(self.current_state, reward[self.models.option_index])
        if len(self.proxy_chain) > 1:
            state, base_state, done, action_list = self.proxy_chain[-1].step(action, model=True, action_list = [action] + action_list)
        else:
            raw_state, factored_state, done = self.proxy_chain[-1].step(action)
            action_list = [action] + action_list

        if done:
            self.reset_history()
        self.raw_state = (raw_state, factored_state)
        # TODO: implement multiprocessing support
        self.extracted_state = pytorch_model.wrap(self.stateExtractor.get_state(self.raw_state), cuda=self.iscuda).unsqueeze(0)
        self.insert_extracted()
        return self.extracted_state, self.raw_state, done, action_list

    def computeReward(self, rollout, length):
        # TODO: probably doesn't have to be in here
        if rollout.cp_filled:
            states = torch.cat([rollout.changepoint_queue[rollout.changepoint_at+1:], rollout.changepoint_queue[:rollout.changepoint_at+1]], dim=0) # multiple policies training
            actions = torch.cat([rollout.changepoint_action_queue[rollout.changepoint_at+1:], rollout.changepoint_action_queue[:rollout.changepoint_at+1]], dim=0)
        else:
            states = rollout.changepoint_queue[:rollout.changepoint_at+1] # multiple policies training
            actions = rollout.changepoint_action_queue[:rollout.changepoint_at+1]
        rewards = []
        # print(states)
        for reward_fn in self.reward_fns:
            rwd = reward_fn.compute_reward(states,actions)
            if len(rwd) < length: #queue not yet filled enough
                ext = torch.zeros((length - len(rwd), )).cuda()
                # print(ext.shape)
                rwd = torch.cat([ext, rwd], dim = 0)
            # print(rwd.shape, length)
            rewards.append(rwd)
        # print(states, rollout.extracted_state)
        # print(torch.stack(rewards, dim=0)[:,-length:].shape)
        return torch.stack(rewards, dim=0)[:,-length:]

    def changepoint_state(self, raw_state):
        return self.reward_fns[0].get_trajectories(raw_state).cuda()
