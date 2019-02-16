import numpy as np
import copy
from ReinforcementLearning.models import pytorch_model
from file_management import get_edge, get_individual_data
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
        self.head, self.tail = get_edge(args.train_edge)
        self.model = model
        self.cuda = args.cuda
        self.traj_dim = 2 #TODO: the dimension of the input trajectory is currently pre-set at 2, the dim of a location. Once we figure out dynamic setting, this can change

    def compute_reward(self, states, actions):
        '''
        takes in states, actions in format: [num in batch (sequential), dim of state/action], there is one more state than action
        for state, action, nextstate
        returns rewards in format: [num in batch, 1]
        '''
        pass

    def get_trajectories(self, full_states):
        # print(self.head)
        obj_dumps = [s[1] for s in full_states]
        trajectory = get_individual_data(self.head, obj_dumps, pos_val_hash=1)
        # TODO: automatically determine if correlate pos_val_hash is 1 or 2
        # TODO: multiple tail support
        if self.tail[0] == "Action":
            # print(obj_dumps, self.tail[0])
            merged = trajectory
            # correlate_trajectory = get_individual_data(self.tail[0], obj_dumps, pos_val_hash=2)
        else:
            correlate_trajectory = get_individual_data(self.tail[0], obj_dumps, pos_val_hash=1)
            merged = np.concatenate([trajectory, correlate_trajectory], axis=1)
        return pytorch_model.wrap(merged).cuda()


class ChangepointDetectionReward(ChangepointReward):
    def __init__(self, model, args, desired_mode):
        super(ChangepointDetectionReward, self).__init__(model, args)
        # self.traj_dim = args.traj_dim
        self.desired_mode = desired_mode
        self.seg_reward = args.segment

    def compute_reward(self, states, actions):
        trajectory = pytorch_model.unwrap(states[:-1,:self.traj_dim])
        saliency_trajectory = pytorch_model.unwrap(states[:-1,self.traj_dim:])
        # print("states shape", trajectory.shape, saliency_trajectory.shape)
        assignments, cps = self.model.get_mode(trajectory, saliency_trajectory)
        rewards = []
        for asmt in assignments:
            if asmt == self.desired_mode:
                rewards.append(1)
            else:
                rewards.append(0)
        rewards.append(0) # match the number of changepoints
        full_rewards = []
        lcp = 0
        lr = 0
        cps.append(len(trajectory))
        # print(cps, rewards)
        for cp, r in zip(cps, rewards):
            if self.seg_reward: # reward copied over all time steps
                full_rewards += [r] * (cp - lcp)
            else:
                if r == 1 and cp == 0:
                    r = 0
                full_rewards +=  [0] * (cp-lcp-1) + [r]
            lcp = cp
            lr = r
        # print(rewards, cps, full_rewards)
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
        self.max_dev = .2 # TODO: this doesn't need to be hardcoded

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
        self.markovModel = model

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
        probs = torch.sum(probs, dim=2).squeeze()
        # print(list(zip(pairs[:,0].squeeze(), n_traj.squeeze(), t_traj.squeeze()))[-1:], probs[-3:])
        return probs

    def compute_reward(self, states, actions):
        probs = self.compute_fit(states)
        reward = torch.ones(probs.size()) * -1
        reward[probs <= self.max_dev] = 2
        # print(states, probs, self.max_dev)
        # print(reward)
        # error
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

reward_forms = {"changepoint": ChangepointDetectionReward, "markov": ChangepointMarkovReward}