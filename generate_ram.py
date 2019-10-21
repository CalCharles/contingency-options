from AtariEnvironments.focus_atari import FocusAtariEnvironment
from SelfBreakout.breakout_screen import RandomPolicy, RotatePolicy
from ObjectRecognition.model import (
    ModelFocusCNN, ModelCollectionDAG,
    load_param, util)
import json, sys, cv2, torch
from Models.models import pytorch_model
from AtariEnvironments.ram_atari import AtariRAMEnvironment
import numpy as np

if __name__ == '__main__':
    torch.cuda.set_device(1)
    screen = AtariRAMEnvironment("Breakout-ramNoFrameskip-v0", 1, 0, sys.argv[1])
    screen.set_save(0, sys.argv[1], -1)
    policy = RotatePolicy(4, 3)
    # policy = BouncePolicy(4)
    last_raw_state = None
    for i in range(200000):
        action = policy.act(screen)
        raw_state, factor_state, done = screen.step(pytorch_model.wrap(action))
        if last_raw_state is not None:
            print(raw_state[np.where(last_raw_state - raw_state != 0)[0]], np.where(last_raw_state - raw_state != 0)[0], action)
        last_raw_state = raw_state
        # raw_state[int(factor_state['Paddle'][0][0]), :] = 255
        # raw_state[:, int(factor_state['Paddle'][0][1])] = 255
        # cv2.imshow('frame',raw_state)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    print("done")

