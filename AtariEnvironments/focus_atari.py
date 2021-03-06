import os, time
from SelfBreakout.breakout_screen import Screen
from Environments.environment_specification import RawEnvironment
from file_management import get_edge
from Models.models import pytorch_model
import numpy as np
import imageio as imio
import torch, cv2
from collections import deque


import gym
from gym.spaces.box import Box

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

try:
    import pybullet_envs
except ImportError:
    pass

class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 84, 84])

    def _observation(self, observation):
        return observation.transpose(2, 0, 1)

    def observation(self, observation):
        return self._observation(observation)

def make_env(env_id, seed, rank, log_dir):
    def _thunk():
        env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)
        env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
        if is_atari:
            env = wrap_deepmind(env)
            env = WrapPyTorch(env)
        return env

    return _thunk

class FocusAtariEnvironment(RawEnvironment):
    '''
    generates the necessary components from the atari environment, including the object dictionary and other components
    '''
    def __init__(self, focus_model, env_id, seed, rank, log_dir):
        try:
            os.makedirs(log_dir)
        except OSError:
            pass
        self.screen_name = (env_id, seed, rank, log_dir)
        self.screen = SubprocVecEnv([make_env(env_id, seed, rank, log_dir)])
        self.num_actions = self.screen.action_space.n
        self.itr = 0
        self.save_path = ""
        self.use_focus = focus_model is None
        self.focus_model = focus_model
        if self.focus_model is not None:
            self.focus_model = focus_model.cuda()
        self.factor_state = None
        self.reward = 0
        self.current_raw = np.squeeze(self.screen.reset())
        self.current_action = [[0.0,0.0], (float(0),)]
        self.episode_rewards = deque(maxlen=10)
        # self.focus_model.cuda()

    def load_new_screen(self):
        self.screen = SubprocVecEnv([make_env(*self.screen_name)])

    def set_save(self, itr, save_dir, recycle, all_dir = ""):
        self.save_path=save_dir
        self.itr = itr
        self.recycle = recycle
        self.all_dir = all_dir
        try:
            os.makedirs(save_dir)
        except OSError:
            pass

    def step(self, action):
        # TODO: action is tensor, might not be safe assumption
        # t = time.time()
        uaction = pytorch_model.unwrap(action.long())
        raw_state, reward, done, infos = self.screen.step([uaction])
        for info in infos:
            if 'episode' in info.keys():
                self.episode_rewards.append(info['episode']['r'])

        # a = time.time()
        # print("screen step", a - t)
        raw_state = np.squeeze(raw_state)
        # raw_state[:10,:] = 0.0
        self.current_raw = raw_state
        factor_state = {'Action': [[0.0,0.0], (float(uaction),)]}
        self.current_action = action
        self.reward = reward[0]
        # cv2.imshow('frame',raw_state)
        # if cv2.waitKey(10000) & 0xFF == ord('q'):
        #     pass

        if self.focus_model is not None:
            factor_state = self.focus_model.forward(pytorch_model.wrap(raw_state.astype(float) / 255.0, cuda=True).unsqueeze(0).unsqueeze(0), ret_numpy=True)
            # t = time.time()
            # print("model step", t - a)
            for key in factor_state.keys():
                factor_state[key] *= 84
                factor_state[key] = (np.squeeze(factor_state[key]), (1.0,))
        self.factor_state = factor_state
        self.last_action = uaction

        # 
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
        print(pval[:-2])
        raw_state = cv2.resize(raw_state, (336, 336))
        cv2.imshow('frame',raw_state)
        if cv2.waitKey(1) & 0xFF == ord(' ') & 0xFF == ord('c'):
            pass

        # logging
        if len(self.save_path) > 0:
            if self.recycle > 0:
                state_path = os.path.join(self.save_path, str((self.itr % self.recycle)//2000))
                count = self.itr % self.recycle
            else:
                state_path = os.path.join(self.save_path, str(self.itr//2000))
                count = self.itr
            try:
                os.makedirs(state_path)
            except OSError:
                pass
            if self.itr != 0:
                object_dumps = open(os.path.join(self.save_path, "focus_dumps.txt"), 'a')
            else:
                object_dumps = open(os.path.join(self.save_path, "focus_dumps.txt"), 'w') # create file if it does not exist
            for key in factor_state.keys():
                writeable = list(factor_state[key][0]) + list(factor_state[key][1])
                object_dumps.write(key + ":" + " ".join([str(fs) for fs in writeable]) + "\t") # TODO: attributes are limited to single floats
            object_dumps.write("\n") # TODO: recycling does not stop object dumping
            
            imio.imsave(os.path.join(state_path, "state" + str(count % 2000) + ".png"), self.current_raw)
            self.itr += 1
        # print("elapsed ", time.time() - t)
        return raw_state, factor_state, done

    def getState(self):
        raw_state = self.current_raw
        factor_state = {'Action': self.current_action}
        if self.factor_state is None:
            if self.focus_model is not None:
                factor_state = self.focus_model.forward(pytorch_model.wrap(raw_state, cuda=True).unsqueeze(0).unsqueeze(0), ret_numpy=True)
                for key in factor_state.keys():
                    factor_state[key] *= 84
                    factor_state[key] = (np.squeeze(factor_state[key]), (1.0,))
            self.factor_state = factor_state
        else:
            factor_state = self.factor_state
        return raw_state, factor_state


