import numpy as np
import copy
from ReinforcementLearning.models import pytorch_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ChangepointReward():
    def __init__(self, model,args):
        '''
        model is a changepoint model
        '''
        self.name = args.train_edge
        self.model = model
        self.cuda = args.cuda
        self.traj_dim = 2 #TODO: the dimension of the input trajectory should be determined by external factors

    def compute_reward(self, states, actions):
        '''
        takes in states, actions in format: [num in batch (sequential), dim of state/action], there is one more state than action
        for state, action, nextstate
        returns rewards in format: [num in batch, 1]
        '''
        pass

class ChangepointDetectionReward(ChangepointReward):
    def __init__(self, model, args, desired_mode):
        super(ChangepointDetectionReward, self).__init__(model, args)
        # self.traj_dim = args.traj_dim
        self.desired_mode = desired_mode
        self.seg_reward = args.segment

    def compute_reward(self, states, actions):
        trajectory = states[:-1,:self.traj_dim]
        saliency_trajectory = states[:-1,self.traj_dim:]
        assignments, cps = self.model.get_mode(trajectory, saliency_trajectory)
        rewards = []
        for asmt in assignments:
            if asmt == self.desired_mode:
                rewards.append(1)
        full_rewards = []
        lcp = 0
        for cp, r in zip(changepoints, rewards):
            if self.seg_reward: # reward copied over all time steps
                full_rewards += [r] * cp - lcp
            else:
                full_rewards += [r] + [0] * (cp-lcp-1)
        return pytorch_model.wrap(np.array(full_rewards), cuda=self.cuda)

class ChangepointMarkovReward(ChangepointReward):
    def __init__(self, model, args, desired_mode):
        super(ChangepointMarkovReward, self).__init__(model, args)
        # self.traj_dim = args.traj_dim
        self.desired_mode = desired_mode
        self.seg_reward = args.segment
        self.hist = args.num_stack
        self.lr = args.lr
        self.eps = args.eps
        self.betas = args.betas
        self.weight_decay = args.weight_decay

    def form_batch(self, data):
        ''' 
        data as sequential data: [number in traj, dim]
        '''
        last_dpt = []
        data_points = []
        for data_pt in data:
            if len(last_dpt) == self.hist - 1:
                data_points.append((last_dpt, copy.deepcopy(last_dpt[1:]) + [copy.deepcopy(data_pt)]))
                last_dpt = copy.deepcopy(last_dpt)
                last_dpt.pop(0)
                last_dpt.append(data_pt)
            else:
                last_dpt.append(copy.deepcopy(data_pt))
        return data_points

    def pytorch_form_batch(self, data):
        ''' 
        data as sequential data: [number in traj, dim]
        '''
        last_dpt = []
        data_points = []
        for data_pt in data:
            if len(last_dpt) == self.hist - 1:
                data_points.append(torch.stack((torch.stack(last_dpt), torch.stack(copy.deepcopy(last_dpt[1:]) + [copy.deepcopy(data_pt)]))))
                last_dpt = copy.deepcopy(last_dpt)
                last_dpt.pop(0)
                last_dpt.append(data_pt)
            else:
                last_dpt.append(copy.deepcopy(data_pt))
        return torch.stack(data_points)

    def generate_training_set(self, states, models, changepoints, match=False, window = -1):
        trajectory = states[:,:self.traj_dim]
        saliency_trajectory = states[:,self.traj_dim:]

        # trajectory = states[:-1,:self.traj_dim]
        # saliency_trajectory = states[:-1,self.traj_dim:]
        assignments, changepoints = self.model.get_mode(trajectory, saliency_trajectory, models, changepoints)
        self.min = np.min(trajectory, axis = 0)
        self.max = np.max(trajectory, axis = 0)
        lcp, cp, ncp = changepoints[0], changepoints[1], changepoints[2]
        asmts = []
        for i in range(3, len(changepoints)-1):
            # print(assignments[i-1],trajectory[lcp:cp+1].squeeze())
            asmts.append((assignments[i-3], trajectory[lcp:cp+1], trajectory[cp+1:ncp], saliency_trajectory[lcp:cp+1], saliency_trajectory[cp+1:ncp]))
            lcp, cp, ncp = cp, ncp, changepoints[i]
        asmts.append((assignments[i-2], trajectory[lcp:cp+1], trajectory[cp+1:ncp], saliency_trajectory[lcp:cp+1], saliency_trajectory[cp+1:ncp]))
        if ncp != len(trajectory):
            lcp, cp, ncp = cp, ncp, len(trajectory)
            asmts.append((assignments[i-1], trajectory[lcp:cp+1], trajectory[cp+1:ncp], saliency_trajectory[lcp:cp+1], saliency_trajectory[cp+1:ncp]))
        self.modes = list(range(self.model.determiner.num_mappings))
        mode_data = {m: [] for m in range(self.model.determiner.num_mappings)}
        for asmt, databefore, dataafter, corrbefore, corrafter in asmts:
            if window < 0:
                data_use = databefore
                if match:
                    other_data = corrbefore
                    # print(data_use.shape, other_data.shape)
                    data_use = np.concatenate((data_use, other_data), axis= 1)
            else:
                data_use = np.concatenate((databefore[-window:], dataafter[:window+1]), axis=0)
                if match:
                    other_data = corrbefore[-window:] + corrafter[:window]
                    data_use = np.concatenate((data_use, other_data), axis= 0)
            if asmt != -1:
                mode_data[asmt] += self.form_batch(data_use)
        total = 0
        for asmt in mode_data.keys():
            mode_data[asmt] = pytorch_model.wrap(mode_data[asmt])
            total += len(mode_data)
        self.pairs = mode_data
        arr = [v.squeeze() for v in self.pairs.values()]
        return total

    def train_rewards(self, epochs_train, save_dir = ""):
        print("###Training rewards###")
        m = self.desired_mode
        print("Training Model", m)
        model = LDSlearner(self.traj_dim, m, self.hist, self.name)
        model.cuda()
        optimizer = optim.Adam(model.parameters(), self.lr, eps=self.eps, betas=self.betas, weight_decay=self.weight_decay)
        avg_err = 0.0
        for i in range(1, epochs_train+1):
            batch = self.pairs[m][np.random.choice(range(len(self.pairs[m])), size = 10)]
            err = model.compute_error(batch).abs().mean()
            i_eq = i % 100 + 1
            avg_err = err/i_eq + avg_err * ((i_eq-1)/i_eq)
            optimizer.zero_grad()
            err.backward()
            optimizer.step()
            if i % 100 == 0:
                print(i, epochs_train, "average error: ", avg_err)
                avg_err = 0
        model.min = self.min
        model.max = self.max
        self.model = model
        var = torch.var(model.compute_error(self.pairs[m]), dim=0)
        var[var < .3] = .3
        model.variance = var

    def compute_fit(self, traj):
        '''
        traj: a trajectory as [num in trajectory, state dim]
        '''
        trajs = []
        tlen = len(traj)
        pairs = self.pytorch_form_batch(traj)
        n_traj = self.markovModel(pairs[:,0])
        t_traj = pairs[:,1]
        probs = self.markovModel.compute_prob(n_traj, t_traj)
        probs = torch.sum(probs, dim=2)
        highprob = torch.argmin(probs, dim=0)
        assignments = []
        for i in highprob:
            assignments.append(i.squeeze()) # not differentiable for now 
        return torch.stack(assignments)

    def compute_reward(self, states, actions):
        assignments = self.compute_fit(states)
        reward = torch.zeros(assignments.size())
        vals = (assignments - self.current_desired_mode).abs()
        reward[vals == 0] = 1
        return reward

class LDSlearner(nn.Module):
    def __init__(self, dim, mode, hist, name):
        super(LDSlearner, self).__init__()
        self.name = name
        self.As = nn.ModuleList([nn.Linear(dim, dim) for _ in range(hist-1)])
        self.dim = dim # dimension of a single state
        self.mode = mode
        self.hist = hist # number of states in the reward function
        self.variance = pytorch_model.wrap([-1 for i in range(dim)])
        self.is_cuda = False

    def forward(self, x):
        '''
        input of the form: [batch size, num stack, dim]
        '''
        # print (x.shape)
        xs = []
        for i, A in enumerate(self.As):
            xs.append(A(x[:, i]))
        # print (xs)
        return torch.stack(xs, dim=1)

    def cuda(self):
        self.is_cuda = True
        self.As.cuda()

    def compute_prob(self, predstep, nextstep):
        # print("variance", self.variance)
        diff = ((predstep - nextstep).abs())/self.variance
        # print("diff",torch.cat([predstep[0:100], nextstep[0:100], diff[0:100]], dim=1))
        # print(diff.size(), self.variance.size())
        # diff = (2*self.variance)/ (diff + 1)
        return diff

    def compute_error(self, inputs):
        # print ("err", self.forward(inputs[:, 0]) , inputs[:, 0], inputs[:, 1], inputs.shape)
        return self.forward(inputs[:, 0]) - inputs[:, 1]

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
                rewards.append(2)
            else:
                rewards.append(-1)
        return pytorch_model.wrap(rewards, cuda=True)

reward_forms = {"changepoint": ChangepointDetectionReward, "markov": ChangepointMarkovReward}