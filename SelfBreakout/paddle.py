import os
from SelfBreakout.breakout_screen import Screen
from Environments.environment_specification import RawEnvironment
from file_management import get_edge

class Paddle(RawEnvironment):
    '''
    A fake environment that pretends that the paddle partion has been solved, gives three actions that produce
    desired behavior
    '''
    def __init__(self, frameskip = 1):
        self.num_actions = 3
        self.itr = 0
        self.save_path = ""
        self.screen = Screen(frameskip=frameskip)
        self.reward= 0
        self.episode_rewards = self.screen.episode_rewards

    def set_save(self, itr, save_dir, recycle, all_dir=""):
        self.save_path=save_dir
        self.itr = itr
        self.recycle = recycle
        self.screen.save_path=save_dir
        self.screen.itr = itr
        self.screen.recycle = recycle
        self.all_dir = all_dir

        try:
            os.makedirs(save_dir)
        except OSError:
            pass

    def step(self, action):
        # TODO: action is tenor, might not be safe assumption
        action = action.clone()
        if action == 1:
            action[0] = 2
        elif action == 2:
            action[0] = 3
        raw_state, factor_state, done = self.screen.step(action, render=True)
        self.reward = self.screen.reward
        if factor_state["Action"][1][0] < 2:
            factor_state["Action"] = (factor_state["Action"][0], 0)
        elif factor_state["Action"][1][0] == 2:
            factor_state["Action"] = (factor_state["Action"][0], 1)
        elif factor_state["Action"][1][0] == 3:
            factor_state["Action"] = (factor_state["Action"][0], 2)
        return raw_state, factor_state, done

    def getState(self):
        raw_state, factor_state = self.screen.getState()
        if factor_state["Action"][1][0] < 2:
            factor_state["Action"] = (factor_state["Action"][0], 0)
        elif factor_state["Action"][1][0] == 2:
            factor_state["Action"] = (factor_state["Action"][0], 1)
        elif factor_state["Action"][1][0] == 3:
            factor_state["Action"] = (factor_state["Action"][0], 2)
        return raw_state, factor_state


