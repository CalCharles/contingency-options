import numpy as np
import os, cv2, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import imageio as imio
from Models.models import pytorch_model, models
from collections import deque
from RewardFunctions.changepointReward import ChangepointDetectionReward



class RawEnvironment():
    def __init__(self):
        self.num_actions = None # this must be defined
        self.itr = 0 # this is used for saving, and is set externally
        self.save_path = "" # save dir also is set externally
        self.episode_rewards = deque(maxlen=10)
        self.all_dir = ""
        self.name = "ABSTRACT_BASE"

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

    def set_save(self, itr, save_dir, recycle, all_dir=""):
        self.save_path=save_dir
        self.itr = itr
        self.recycle = recycle
        self.all_dir = all_dir
        try:
            os.makedirs(save_dir)
        except OSError:
            pass


class ChainMDP(RawEnvironment):
    def __init__(self, num_states, num_dims, terminal_state):
        super(ChainMDP, self).__init__()
        self.minmax = (np.array([0 for i in range(num_dims)]),np.array([num_states for i in range(num_dims)]))
        self.num_states = num_states
        self.initial_state = np.array([self.num_states//2 for i in range(num_dims)])
        self.current_state = self.initial_state
        self.num_actions = num_dims * 2 + 1
        self.num_dims = num_dims
        self.terminal_state = terminal_state
        self.reward = 0

    def step(self, action):
        action = int(pytorch_model.unwrap(action[0]))
        # print("action", action)
        if action == 0: # noop
            pass
        elif action % 2 == 1:
            v = self.current_state[(action-1) // 2]
            if v != 0:
                ncs = self.current_state.copy()
                ncs[(action-1) // 2] -= 1
                self.current_state = ncs
        elif action % 2 == 0:
            v = self.current_state[(action-1) // 2]
            if v != self.num_states - 1:
                ncs = self.current_state.copy()
                ncs[(action-1) // 2] += 1
                self.current_state = ncs
        done = False
        if self.terminal_state:
            done = True
            for i in range(self.num_dims):
                done = done and ((self.current_state[i] == self.num_states - 1) or (self.current_state[i] == 0))
            if done:
                self.current_state =self.initial_state
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
        if self.itr % 100 == 0:
            self.current_state =self.initial_state
            done = True

        # if done:
        #     self.current_state[0] = 0
        return self.current_state, {"chain": (self.current_state, 1)}, done

    def getState(self):
        return self.current_state, {"chain": (self.current_state, 1)}


class ProxyEnvironment():
    def __init__(self, name= "0"):
        '''
        create a dummy placeholder
        '''
        self.name = name
        self.proxy_chain = [None]

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
        for i, env in enumerate(self.proxy_chain[1:]):
            print(env.name, [e.name for e in self.proxy_chain], [e.name for e in self.proxy_chain[:i+1]])
            env.proxy_chain = self.proxy_chain[:i + 1] 
            env.iscuda = args.cuda # TODO: changes to this probably will cause breaking
            if args.cuda:
                current_cuda = torch.device('cuda:' + str(args.gpu))
                env.changepoint_queue, env.changepoint_action_queue, env.changepoint_resp_queue = env.changepoint_queue.to(current_cuda), env.changepoint_action_queue.to(current_cuda), env.changepoint_resp_queue.to(current_cuda)
                env.extracted_state, env.resp, env.current_state, env.current_resp = env.extracted_state.to(current_cuda), env.resp.to(current_cuda), env.current_state.to(current_cuda), env.current_resp.to(current_cuda)
        self.reward_fns = reward_fns
        self.stateExtractor = state_get
        self.iscuda = args.cuda

        self.swap_form = args.swap_form
        self.swap = True
        self.delayed_swap = False
        if self.swap_form == 'reward':
            self.delayed_swap = True
        self.current_action = 0
        self.num_hist = args.num_stack
        self.state_size = self.stateExtractor.shape
        # self.action_size = self.stateExtractor.action_num
        self.lag_num = args.lag_num
        self.behavior_policy = behavior_policy
        self.reset_history()
        print(proxy_chain[0].getState())
        print(self.stateExtractor.get_state(proxy_chain[0].getState())[0].shape)
        self.state_shape = self.stateExtractor.get_state(proxy_chain[0].getState())[0].shape
        fs, fr = self.stateExtractor.get_state(proxy_chain[0].getState())
        self.extracted_state = pytorch_model.wrap(fs, cuda=args.cuda).detach()
        self.resp = pytorch_model.wrap(fr, cuda=args.cuda)
        self.resp_len = len(self.stateExtractor.fnames)
        self.insert_extracted()
        self.parameterized_option = args.parameterized_option
        print(self.current_state.shape, self.state_size)

        self.changepoint_queue_len = args.changepoint_queue_len
        print(self.reward_fns, proxy_chain)
        self.changepoint_shape = self.reward_fns[0].get_trajectories([proxy_chain[0].getState()]).shape[1:]
        self.changepoint_queue = torch.zeros(self.changepoint_queue_len, *self.changepoint_shape).detach()
        self.changepoint_resp_queue = torch.zeros(self.changepoint_queue_len, *self.resp.shape).detach()
        self.changepoint_action_queue = torch.zeros(self.changepoint_queue_len, 1).long().detach() # TODO: add responsibility to changepoints
        self.changepoint_filled = 0
        if self.iscuda:
            self.changepoint_queue = self.changepoint_queue.cuda()
            self.changepoint_action_queue = self.changepoint_action_queue.cuda()
            self.changepoint_resp_queue = self.changepoint_resp_queue.cuda()

        self.changepoint_at = 0
        self.cp_filled = False

        print("num_reward_functions", len(self.reward_fns))

    def get_next_parameter(self):
        if self.parameterized_option == 0:
            return len(self.reward_fns)
        else:
            return self.reward_fns[0].get_possible_parameters(self.changepoint_queue[self.changepoint_filled-1])

    def get_names(self):
        if len(self.proxy_chain) > 1:
            return self.proxy_chain[-1].get_names() + [self.name]
        else: 
            return [self.name]

    def set_save(self, itr, save_dir, recycle):
        '''
        sets the files for saving the action data
        '''
        self.save_path=save_dir
        self.itr = itr
        self.recycle = recycle
        if len(self.save_path) > 0:
            try:
                os.makedirs(save_dir)
            except OSError:
                pass
            self.save_files = []
            for name in self.get_names():
                print(self.get_names())
                if name.find("->") != -1:
                    name = name.split("->")[0] # use the base
                f = open(os.path.join(self.save_path, name + "_actions.txt"), 'w')
                print(os.path.join(self.save_path, name + "_actions.txt"))
                self.save_files.append(f)

    def save_actions(self, action_list):
        if len(self.save_path) > 0:
            for i, action in enumerate(action_list):
                self.save_files[i].write(str(int(pytorch_model.unwrap(action.squeeze()))) + '\n')

    def close_files(self):
        if len(self.save_path) > 0:
            for f in self.save_files:
                f.close()

    def set_models(self, models):
        self.models = models

    def duplicate(self, args):
        if len(self.reward_fns) > len(self.models.models) or args.model_form == 'adjust' or (args.model_form == 'population' and type(self.models.models[0]) != models['population']):
            # self.reward_fns[0].parameter_minmax = None
            self.models.duplicate(len(self.reward_fns), args, self.stateExtractor, self.action_size, self.reward_fns[0].parameter_minmax)

    def set_proxy_chain(self, proxy_chain):
        self.proxy_chain = proxy_chain

    def set_test(self):
        self.behavior_policy.set_test()

    def set_checkpoint(self):
        '''
        sets a save point that can be returned to later
        '''
        self.saved_current_state = self.current_state.clone().detach()
        self.saved_extracted_state = self.extracted_state.clone().detach()
        self.proxy_chain[-1].set_checkpoint()

    def load_checkpoint(self):
        self.current_state = self.saved_current_state
        self.extracted_state = self.saved_extracted_state
        self.proxy_chain[-1].set_checkpoint()

    def reset_history(self):
        self.current_state = pytorch_model.wrap(np.zeros((self.num_hist * int(np.prod(self.state_size)), )), cuda = self.iscuda)
        self.current_resp = pytorch_model.wrap([[0 for i in range(len(self.stateExtractor.fnames))] for _ in range(self.num_hist)], cuda = self.iscuda).flatten()
        # TODO: add multi-process code someday

    def insert_extracted(self):
        '''
        self.current_state has history, and is of shape: [hist len * state size] TODO: [batch/processor number, hist len * state size]
        '''
        shape_dim0 = self.num_hist # make sure this is 1 if no history is to be used
        state_size = int(np.prod(self.state_size))
        # print(self.extracted_state.shape, self.current_state.shape, self.current_resp, self.resp)
        if self.num_hist > 1:
            self.current_state[:(shape_dim0-1)*state_size] = self.current_state[-(shape_dim0-1)*state_size:].detach()
            self.current_resp[:-self.resp_len] = self.current_resp[self.resp_len:].detach()
        try:
            len(self.state_shape) < 2
        except AttributeError as e:
            self.state_shape = (2,)
        if len(self.state_shape) < 2: # TODO: a hacked result, try to fix?
            self.current_state[-state_size:] = self.extracted_state.detach() # unsqueeze 0 is for dummy multi-process code
            # print(self.current_resp, self.resp)
            self.current_resp[-self.resp_len:] = self.resp
        else:
            self.current_state = self.extracted_state.detach() # unsqueeze 0 is for dummy multi-process code
            # print(self.current_resp, self.resp)
            self.current_resp = self.resp

        return self.current_state.detach()

    def getState(self):
        return self.extracted_state

    def getResp(self):
        return self.resp

    def getHistState(self):
        return self.current_state, self.current_resp

    def step(self, action, model=False, action_list=[]):
        '''
        steps the true environment. The last environment in the proxy chain is the true environment,
        and has a different step function.
        raw_state is the tuple (raw_state, factor_state)
        model determines if action is a model 
        extracted state is the proxy extracted state, raw state is the full raw state (raw_state, factored state),
        done defines if a trajectory ends, action_list is all the actions taken by all the options in the chain
        '''
        # print(self.name)
        if self.swap:
            self.current_action = action
        else:
            action = self.current_action
        if model:
            if self.swap:
                old_action = action
                values, log_probs, probs, Q_vals = self.models.determine_action(self.current_state.unsqueeze(0), self.current_resp.unsqueeze(0))
                vals, action_probs, log_probs, Q_vs = self.models.get_action(values, probs, log_probs, Q_vals, index = int(action))
                action = self.behavior_policy.take_action(action_probs, Q_vs)
                self.current_action = action
                # print(self.name, action, old_action, self.stateExtractor.names, self.stateExtractor.fnames, self.current_state, self.current_resp)
            else:
                action = self.current_action
            # if issubclass(self.models.currentModel(), DopamineModel): 
            #     reward = self.computeReward(rollout, 1)
            #     action = self.models.currentModel().forward(self.current_state, reward[self.models.option_index])
        # print([p.name for p in self.proxy_chain])
        if len(self.proxy_chain) > 1:
            state, base_state, resp, done, action_list = self.proxy_chain[-1].step(action, model=True, action_list = [action] + action_list)
            raw_state, factored_state = base_state
        else:
            raw_state, factored_state, done = self.proxy_chain[-1].step(action)
            action_list = [action] + action_list

        if done:
            self.reset_history()
        # print(action_list)
        # bpos = factored_state["Ball"][0]
        # ppos = factored_state["Paddle"][0]
        # raw_state[int(bpos[0]), :] = 255
        # raw_state[:, int(bpos[1])] = 255
        # raw_state[int(ppos[0]), :] = 255
        # raw_state[:, int(ppos[1])] = 255
        # cv2.imshow("frame", raw_state)
        # cv2.waitKey(5)
        # print(bpos)
        self.raw_state = (raw_state, factored_state)
        # TODO: implement multiprocessing support
        state, resp = self.stateExtractor.get_state(self.raw_state)
        self.resp = pytorch_model.wrap(resp, cuda=self.iscuda)
        self.extracted_state = pytorch_model.wrap(state, cuda=self.iscuda).unsqueeze(0)
        self.insert_extracted()
        if not model: # at the top level
            self.save_actions(action_list)
        self.cp_state = self.changepoint_state([self.raw_state])
        self.insert_changepoint_queue(self.cp_state, pytorch_model.wrap(action, cuda=self.iscuda).detach(), pytorch_model.wrap(resp, cuda=self.iscuda).detach())
        if self.delayed_swap and self.swap: # we just swapped and we are using delayed swapping
            self.swap = False
        return self.extracted_state, self.raw_state, self.resp, done, action_list

    def cpu(self):
        self.extracted_state = self.extracted_state.cpu()
        self.resp = self.resp.cpu()
        self.changepoint_queue = self.changepoint_queue.cpu()
        self.changepoint_action_queue = self.changepoint_action_queue.cpu()
        self.changepoint_resp_queue = self.changepoint_resp_queue.cpu()
        self.models = self.models.cpu()

    def cuda(self):
        self.extracted_state = self.extracted_state.cuda()
        self.resp = self.resp.cuda()
        self.changepoint_queue = self.changepoint_queue.cuda()
        self.changepoint_action_queue = self.changepoint_action_queue.cuda()
        self.changepoint_resp_queue = self.changepoint_resp_queue.cuda()
        self.models = self.models.cuda()

    def step_dope(self, action, rollout, model=False, action_list=[]):
        '''
        steps the true environment, using dopamine models. The last environment in the proxy chain is the true environment,
        and has a different step function (performs model updates and most state saving inside dopamine)
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
        state, resp = self.stateExtractor.get_state(self.raw_state)
        self.extracted_state = pytorch_model.wrap(state, cuda=self.iscuda).unsqueeze(0)
        self.insert_extracted()
        self.insert_changepoint_queue(self.cp_state, pytorch_model.wrap(action, cuda=self.iscuda), pytorch_model.wrap(resp, cuda=self.iscuda))
        return self.extracted_state, self.raw_state, done, action_list

    def computeReward(self, length):
        # TODO: probably doesn't have to be in here
        states = self.changepoint_queue[:self.changepoint_filled] # multiple policies training
        actions = self.changepoint_action_queue[:self.changepoint_filled]
        resps = self.changepoint_resp_queue[:self.changepoint_filled]
        rewards = []
        # print(rollout.changepoint_queue.shape)
        # print("raw", states, self.reward_fns[0].rewards)
        # print(self.reward_fns)
        # start=time.time()

        precomputed = None
        if type(self.reward_fns[0]) == ChangepointDetectionReward: # hacky way to precompute changepoints, but improves speed by a lot
            precomputed = self.reward_fns[0].precompute(states,actions,resps)
        
        for reward_fn in self.reward_fns:
            # print(states, actions, resps)
            rwd = reward_fn.compute_reward(states,actions,resps, precomputed = precomputed)
            # xloss = -(states[:,1] - states[:,-1]).abs() * 1e-4
            if len(rwd) < length: # queue not yet filled enough
                ext = torch.zeros((length - len(rwd), )).cuda().detach()
                # print(ext.shape)
                rwd = torch.cat([ext, rwd], dim = 0)
                # xloss = torch.cat([ext, xloss], dim = 0)
            # print(rwd.shape, length)

            # rwd += xloss
            rewards.append(rwd)
        # print(rewards, length, states, actions)
        # print(rewards, states)
        # print(time.time() - start)
        # print(torch.stack(rewards, dim=0)[:,-length:].shape)
        # print(states, rollout.extracted_state)
        # print(length, self.lag_num, self.changepoint_filled)
        # print(states, rewards)
        self.current_rewards = torch.stack(rewards, dim=0)[:,max(self.changepoint_filled-length-self.lag_num, 0):self.changepoint_filled-self.lag_num]
        # print("attached", self.current_rewards, rewards)
        # print(states.shape, self.current_rewards.shape)
        # print(torch.cat([self.current_rewards.transpose(0,1), states[-self.current_rewards.shape[1]:]], dim=1)[:100])
        # print(torch.stack(rewards, dim=0).shape)
        # print(self.current_rewards.shape, self.changepoint_filled-length-self.lag_num,self.changepoint_filled-self.lag_num, torch.stack(rewards, dim=0).shape)
        # error
        # print("cp_queue", self.changepoint_queue[self.changepoint_filled-length-self.lag_num:self.changepoint_filled-self.lag_num])
        return self.current_rewards

    def determineChanged(self, length):
        states = self.changepoint_queue[self.changepoint_filled - length:self.changepoint_filled] # multiple policies training
        actions = self.changepoint_action_queue[self.changepoint_filled - length:self.changepoint_filled]
        resps = self.changepoint_resp_queue[self.changepoint_filled - length:self.changepoint_filled]

        change, state = self.reward_fns[0].determineChanged(states, actions, resps)
        return change, state

    def determine_swaps(self, length, needs_rewards=True):
        if len(self.proxy_chain) > 1:
            self.proxy_chain[-1].determine_swaps(length)
        if self.swap_form == "dense": # TODO: swap based on rewards for all
            self.swap = True
        elif self.swap_form == "reward":
            if needs_rewards:
                self.current_rewards = self.proxy_chain[-1].computeReward(length)
            self.swap = self.current_rewards.sum() > 0.3
            # print(self.name, self.swap, self.current_rewards.shape, len(self.proxy_chain), length)

    def changepoint_state(self, raw_state):
        self.cp_state = self.reward_fns[0].get_trajectories(raw_state)
        if self.iscuda:
            self.cp_state = self.cp_state.cuda().detach()
        return self.cp_state

    def insert_changepoint_queue(self, state, action, resp):
        if self.changepoint_queue_len > 0:
            if self.changepoint_filled == self.changepoint_queue_len:
                self.changepoint_action_queue = self.changepoint_action_queue.roll(-1, 0).detach()
                self.changepoint_resp_queue = self.changepoint_resp_queue.roll(-1, 0).detach()
                self.changepoint_queue = self.changepoint_queue.roll(-1, 0).detach()
            self.changepoint_filled += self.changepoint_filled < self.changepoint_queue_len
            self.changepoint_action_queue[self.changepoint_filled-1].copy_(action.squeeze().detach())
            self.changepoint_resp_queue[self.changepoint_filled-1].copy_(resp.squeeze().detach())
            self.changepoint_queue[self.changepoint_filled-1].copy_(state.squeeze().detach())
            # print(self.changepoint_queue)
