import os, time, cv2
from SelfBreakout.breakout_screen import Screen
from Environments.environment_specification import RawEnvironment
from file_management import get_edge
from Models.models import pytorch_model
import numpy as np

class FocusEnvironment(RawEnvironment):
    '''
    A fake environment that pretends that the paddle partion has been solved, gives three actions that produce
    desired behavior
    '''
    def __init__(self, focus_model, frameskip=1, display=False):
        self.num_actions = 4
        self.itr = 0
        self.save_path = ""
        self.screen = Screen(frameskip=frameskip)
        self.focus_model = focus_model.cuda()
        self.factor_state = None
        self.reward = 0
        self.episode_rewards = self.screen.episode_rewards
        self.display=display
        # self.focus_model.cuda()

    def set_save(self, itr, save_dir, recycle, all_dir=""):
        self.save_path=save_dir
        self.itr = itr
        self.recycle = recycle
        self.screen.save_path=save_dir
        self.screen.itr = itr
        self.screen.recycle = recycle
        self.screen.all_dir = all_dir
        try:
            os.makedirs(save_dir)
        except OSError:
            pass

    def step(self, action):
        # TODO: action is tensor, might not be safe assumption
        t = time.time()
        raw_state, raw_factor_state, done = self.screen.step(action, render=True)
        self.reward = self.screen.reward
        factor_state = self.focus_model.forward(pytorch_model.wrap(raw_state, cuda=False).unsqueeze(0).unsqueeze(0), ret_numpy=True)
        for key in factor_state.keys():
            factor_state[key] *= 84
            factor_state[key] = (np.squeeze(factor_state[key]), (1.0,))
        factor_state['Action'] = raw_factor_state['Action']
        self.factor_state = factor_state

        if self.display:
            rs = raw_state.copy()
            time_dict = factor_state
            pval = ""
            for k in time_dict.keys():
                if k != 'Action' and k != 'Reward':
                    raw_state[int(time_dict[k][0][0]), :] = 255
                    raw_state[:, int(time_dict[k][0][1])] = 255
                if k == 'Action' or k == 'Reward':
                    pval += k + ": " + str(time_dict[k][1]) + ", "
                else:
                    pval += k + ": " + str(time_dict[k][0]) + ", "
            # print(pval[:-2])
            raw_state = cv2.resize(raw_state, (336, 336))
            cv2.imshow('frame',raw_state)
            if cv2.waitKey(1) & 0xFF == ord(' ') & 0xFF == ord('c'):
                pass


        if self.screen.itr != 0:
            object_dumps = open(os.path.join(self.save_path, "focus_dumps.txt"), 'a')
        else:
            object_dumps = open(os.path.join(self.save_path, "focus_dumps.txt"), 'w') # create file if it does not exist
        for key in factor_state.keys():
            writeable = list(factor_state[key][0]) + list(factor_state[key][1])
            object_dumps.write(key + ":" + " ".join([str(fs) for fs in writeable]) + "\t") # TODO: attributes are limited to single floats
        object_dumps.write("\n") # TODO: recycling does not stop object dumping
        # print("elapsed ", time.time() - t)
        return raw_state, factor_state, done

    def getState(self):
        raw_state, raw_factor_state = self.screen.getState()
        if self.factor_state is None:
            factor_state = self.focus_model.forward(pytorch_model.wrap(raw_state, cuda=False).unsqueeze(0).unsqueeze(0), ret_numpy=True)
            for key in factor_state.keys():
                factor_state[key] *= 84
                factor_state[key] = (np.squeeze(factor_state[key]), (1.0,))
            factor_state['Action'] = raw_factor_state['Action']
            self.factor_state = factor_state
        factor_state = self.factor_state
        return raw_state, factor_state


