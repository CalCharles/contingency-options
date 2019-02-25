import os
import numpy as np
import skimage.io as imio
import matplotlib.pyplot as plt
from functools import partial

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from SelfBreakout.breakout_screen import read_obj_dumps, get_individual_data
from ObjectRecognition.gym_wrapper import make_env
import ObjectRecognition.util as util


# Dataset interface
class DatasetInterface:

    """
    get frames in a batch, abstract method specific for game
        if r is given: index l to r
        otherwise: get all indices in l
    """
    def get_frame(self, l, r=None):
        raise NotImplementedError


    # get frame size
    def get_shape(self):
        raise NotImplementedError


    # extract change points from game actions at list of time indices
    def get_changepoint(self):
        raise NotImplementedError


    # get history of actions associated with the frame
    def retrieve_action(self, idxs):
        raise NotImplementedError


    # reset state
    def reset(self):
        raise NotImplementedError


class Dataset(DatasetInterface):

    # extract change points from game actions at list of time indices
    def get_changepoint(self):
        # boolean whether action is changing
        action = self.retrieve_action(np.arange(self.get_shape()[0]))
        action_ischange = util.next_noteq(action)

        # extract changepoints of actions
        action_cp = np.arange(action.shape[0])[action_ischange]

        return action_cp


# breakout saved scene loading, handle block loading
class DatasetSelfBreakout(Dataset):
    TOTAL_STATE = 10000  # TODO: realize this?

    def __init__(self, objdump_path, state_path, *args, **kwargs):
        super(DatasetSelfBreakout, self).__init__()
        obj_dumps = read_obj_dumps(objdump_path)
        self.state_path = state_path
        self.frame_shape = (84, 84)
        self.n_state = kwargs.get('n_state', 1000)
        self.block_size = kwargs.get('block_size', 10000)
        self.binarize = kwargs.get('binarize', None)
        self.frame_l, self.frame_r = 0, -1  # uninitialized frame interval

        self.block_size = min(self.block_size, self.n_state)

        self.actions = np.array(
            get_individual_data('Action', obj_dumps, pos_val_hash=2))
        self.paddle_data = np.array(
            get_individual_data('Paddle', obj_dumps, pos_val_hash=1),
            dtype=int)
        self.ball_data = np.array(
            get_individual_data('Ball', obj_dumps, pos_val_hash=1),
            dtype=int)

        self.reset()
        

    # retrieve selected actions associated with frames
    def retrieve_action(self, idxs):
        return self.actions[idxs]


    # get frame at index l to r
    def get_frame(self, l, r=None):
        # get a frame
        if r is None and not hasattr(l, '__iter__'):
            if l >= self.frame_r:  # move right
                self._load_range(l)
            if l < self.frame_l:  # move left
                self._load_range(l-self.block_size+1)
            return self.frame_buffer[l-self.frame_l]

        # get many indexes
        elif r is None:
            return np.array([self.get_frame(li) for li in l])

        # if still in buffer
        elif self.frame_l <= l and r <= self.frame_r:
            return self.frame_buffer[l-self.frame_l:r-self.frame_l]

        # if out of buffer
        frames = np.zeros((r-l, 1,) + self.frame_shape)
        cur_idx = 0
        while cur_idx+l < r:
            self._load_range(cur_idx+l)
            if self.frame_r > r:  # copy part of buffer
                frames[cur_idx:, ...] = self.frame_buffer[:r-l-cur_idx, ...]
            else:  # copy everthing
                frames[cur_idx:self.frame_r-l, ...] = self.frame_buffer[:, ...]
            cur_idx = self.frame_r-l
        return frames


    # get frame size
    def get_shape(self):
        return (self.n_state, 1,) + self.frame_shape


    # reset state (do nothing)
    def reset(self):
        # to simulate generating new episode
        self.idx_offset = np.random.randint(
            low=0, 
            high=DatasetSelfBreakout.TOTAL_STATE-self.n_state)
        self._load_range(self.frame_l)


    # load batch
    def _load_range(self, l):
        self.frame_l = l
        self.frame_r = l + self.block_size
        self.frame_buffer = np.zeros((self.frame_r-self.frame_l, 1) + self.frame_shape)
        for idx in range(self.frame_l, self.frame_r):
            self.frame_buffer[idx-self.frame_l, :] = self._load_image(idx)

        # binary to simplify image
        if self.binarize:
            self.frame_buffer[self.frame_buffer < self.binarize] = 0.0
            self.frame_buffer[self.frame_buffer >= self.binarize] = 1.0


    # load a scene
    def _load_image(self, idx):
        try:
            img = imio.imread(self._get_image_path(idx), as_gray=True) / 256.0
        except FileNotFoundError:
            img = np.full(self.frame_shape, 0)
        return img


    # image path
    def _get_image_path(self, idx):
        # to simulate generating new episode
        real_idx = idx + self.idx_offset
        return os.path.join(self.state_path, 'state%d.png'%(real_idx))


# access to atari games
class DatasetAtari(Dataset):

    def __init__(self, game_name, Actor, n_state, save_path, *args, **kwargs):

        # create OpenAI Atari
        self.save_path = util.get_dir(save_path)
        self.atari_env = SubprocVecEnv([make_env('BreakoutNoFrameskip-v4', 
                                                 1234, 0, self.save_path)])
        self.frame_shape = self.atari_env.observation_space.shape[-2:]
        self.n_state = n_state
        self.binarize = kwargs.get('binarize', None)

        # actor for auto run
        self.action_space = self.atari_env.action_space.n
        self.actor = Actor(self.action_space)

        # TODO: online generatation
        self._generate_all_states()


    """
    get frames in a batch
        if r is given: index l to r
        otherwise: get all indices in l
    """
    def get_frame(self, l, r=None):
        if r is None:
            if not hasattr(l, '__iter__'):
                return self.frames[l]
            else:
                return self.frames[l]
        else:
            return self.frames[np.arange(l, r)]


    # get frame size
    def get_shape(self):
        return (self.n_state, 1,) + self.frame_shape


    # get history of actions associated with the frame
    def retrieve_action(self, idxs):
        return self.acts[idxs]


    # reset state, generate new states
    def reset(self):
        self._generate_all_states()


    # generate all states
    def _generate_all_states(self):
        self.acts = np.zeros((self.n_state))
        self.frames = np.zeros((self.n_state, 1,) + self.frame_shape)
        state = self.atari_env.reset()
        for i in range(self.n_state):
            # get act from actor and step forward
            act = self.actor.act(state)
            state, reward, done, info = self.atari_env.step([act])

            # record state
            self.acts[i] = act
            self.frames[i, ...] = state[0]

        # feature scaling normalize
        self.frames = (self.frames - np.min(self.frames)) / \
                      (np.max(self.frames) - np.min(self.frames))

        # binary to simplify image
        if self.binarize:
            self.frames[self.frames < self.binarize] = 0.0
            self.frames[self.frames >= self.binarize] = 1.0