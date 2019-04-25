import os, time
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
    def __init__(self, focus_model):
        self.num_actions = 4
        self.itr = 0
        self.save_path = ""
        self.screen = Screen()
        self.focus_model = focus_model
        self.factor_state = None
        self.reward = 0
        # self.focus_model.cuda()

    def set_save(self, itr, save_dir, recycle):
        self.save_path=save_dir
        self.itr = itr
        self.recycle = recycle
        self.screen.save_path=save_dir
        self.screen.itr = itr
        self.screen.recycle = recycle
        try:
            os.makedirs(save_dir)
        except OSError:
            pass

    def step(self, action):
        # TODO: action is tenor, might not be safe assumption
        t = time.time()
        raw_state, raw_factor_state, done = self.screen.step(action, render=True)
        self.reward = self.screen.reward
        factor_state = self.focus_model.forward(pytorch_model.wrap(raw_state, cuda=False).unsqueeze(0).unsqueeze(0), ret_numpy=True)
        for key in factor_state.keys():
            factor_state[key] *= 84
            factor_state[key] = (np.squeeze(factor_state[key]), (1.0,))
        factor_state['Action'] = raw_factor_state['Action']
        self.factor_state = factor_state
        if self.screen.itr != 0:
            object_dumps = open(os.path.join(self.save_path, "focus_dumps.txt"), 'a')
        else:
            object_dumps = open(os.path.join(self.save_path, "focus_dumps.txt"), 'w') # create file if it does not exist
        for key in factor_state.keys():
            object_dumps.write(key + ":" + " ".join([str(fs) for fs in factor_state[key]]) + "\t") # TODO: attributes are limited to single floats
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


