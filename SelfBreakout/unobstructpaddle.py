import os
from SelfBreakout.breakout_screen import Screen
from SelfBreakout.paddle import Paddle
from Environments.environment_specification import RawEnvironment
from file_management import get_edge
import numpy as np

class PaddleNoWalls(Paddle):
    '''
    A fake environment that pretends that the paddle partion has been solved, gives three actions that produce
    desired behavior, and has no blocks (one fake block to prevent over -resetting)
    '''
    def __init__(self):
        super().__init__()
        

    def reset(self):
        self.screen.paddle.nowall = False
        self.screen.render_frame()

    def step(self, action):
        raw_state, factor_state, done = super().step(action)
        if done:
            self.reset()
        raw_state, factor_state = super().getState()
        return raw_state, factor_state, done


