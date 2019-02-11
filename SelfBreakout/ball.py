from SelfBreakout.breakout_screen import Screen
from SelfBreakout.breakout_objects import intersection
import copy

class Ball(RawEnvironment):
    '''
    A fake environment that pretends that the paddle partion has been solved, gives three actions that produce
    desired behavior
    '''
    def __init__(self):
        self.num_actions = 4
        self.itr = 0
        self.save_path = ""
        self.screen = Screen()
        self.internal_screen = copy.deepcopy(screen)

    def step(self, action):
        if action == 1:
            action = 2
        elif action == 2:
            action = 3
        raw_state, factor_state = self.screen.getState()
        ball = factor_state["Ball"][0]
        ball_vel = self.screen.ball.vel
        if ball_vel[0] < 0 or ball[0] > 60: # ball is too far or moving up, so we don't care where it is
            # TODO: follow the ball
        else:
            self.internal_screen = copy.deepcopy(screen)
            while self.internal_screen.ball.pos[0] < 71:
                self.internal_screen.step([0])
            self.objective_location = self.internal_screen.ball.pos[1] + np.random.choice([-1, 0, 1])

        paddle = factor_state["Paddle"][0]
        raw_state, factor_state, done = self.screen.step(action)
        if factor_state["Action"][1] < 2:
            factor_state["Action"][1] = 0
        elif factor_state["Action"][1] == 2:
            factor_state["Action"][1] = 1
        elif factor_state["Action"][1] == 3:
            factor_state["Action"][1] = 2

    def getState(self):
        raw_state, factor_state = self.screen.getState()
        if factor_state["Action"][1] < 2:
            factor_state["Action"][1] = 0
        elif factor_state["Action"][1] == 2:
            factor_state["Action"][1] = 1
        elif factor_state["Action"][1] == 3:
            factor_state["Action"][1] = 2

