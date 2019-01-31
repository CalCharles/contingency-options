
class RawEnvironment():
    def __init__(self):
        self.num_actions = None # this must be defined
        pass

    def step(self, action):
        '''
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

class ChainMDP(RawEnvironment):
    def __init__(self, num_states):
        self.minmax = (0,num_states)
        self.num_states = num_states
        self.current_state = np.array([0])

    def step(self, action):
        if action == 0 and self.current_state[0] > 0:
            self.current_state[0] -= 1
        elif action == 1:
            pass
        elif action == 2 and self.current_state[0] < self.num_states-1:
            self.current_state += 1
        done = self.current_state[0] == self.num_states - 1
        # if done:
        #     self.current_state[0] = 0
        return self.current_state, {"chain": (self.current_state, 1)}, done

    def getState(self):
        return self.current_state


class ProxyEnvironment():
    def __init__(self, args, proxy_chain, reward_fns, state_get):
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
        self.args = args
        self.name = args.unique_id # name should be of form: head_tail

        self.num_hist = args.num_stack
        self.state_size = self.stateExtractor.state_size
        self.action_size = self.stateExtractor.action_size
        self.reset_history()
        self.extracted_state = torch.Tensor(self.stateExtractor.get_state(proxy_chain[0].getState())).cuda()
        self.insert_extracted()

    def set_models(self, models):
        self.models = models

    def set_proxy_chain(self, proxy_chain):
        self.proxy_chain = proxy_chain


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
        self.current_state[:, shape_dim0*state_size-1:] = self.extracted_state # unsqueeze 0 is for dummy multi-process code
        return self.current_state

    def getState(self):
        return self.extracted_state

    def getHistState(self):
        return self.current_state

    def step(self, action, model=False):
        '''
        steps the true environment. The last environment in the proxy chain is the true environment,
        and has a different step function.
        raw_state is the tuple (raw_state, factor_state)
        model determines if action is a model 
        '''
        if model:
            action, values, action_probs, Q_vals = self.models.determine_action(self.current_state, index=action)
        if len(self.proxy_chain) > 1:
            state, base_state, done = self.proxy_chain[-1].step(action, model=True)
        else:
            base_state, done = self.proxy_chain[-1].step(action)
        if done:
            self.reset_history()
        self.base_state = base_state
        # TODO: implement multiprocessing support
        self.extracted_state = pytorch_model.wrap(self.stateExtractor.get_state(base_state), cuda=self.args.cuda).unsqueeze(0)
        self.insert_extracted()
        return self.extracted_state, self.raw_state, done

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
