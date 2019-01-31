import os
import numpy as np
import skimage.io as imio
import matplotlib.pyplot as plt

from selfBreakout.breakout_screen import read_obj_dumps, get_individual_data
from game_environment import GameEnvironment


# curently for handling breakout saved scene loading 
class BreakoutLog(GameEnvironment):

    def __init__(self, objdump_path, state_path, frame_shape=(84, 84), *args, **kwargs):
        super(BreakoutLog, self).__init__()
        obj_dumps = read_obj_dumps(objdump_path)
        self.state_path = state_path
        self.frame_shape = frame_shape
        self.batch_size = kwargs.get('batch_size', 10000)
        self.frame_l, self.frame_r = 1, 0  # uninitialized frame interval

        self.actions = np.array(get_individual_data('Action', obj_dumps, pos_val_hash=2))
        self.paddle_data = np.array(get_individual_data('Paddle', obj_dumps, pos_val_hash=1), dtype=int)
        

    # retrieve selected actions associated with frames
    def retrieve_action(self, idxs):
        return self.actions[idxs]


    # get frame at index idx
    def get_frame(self, idx):
        if idx >= self.frame_r:  # move right
            self.load_range(idx)
        if idx < self.frame_l:  # move left
            self.load_range(idx-self.batch_size+1)
        return self.frame_buffer[idx-self.frame_l]


    # get frame at index l to r
    def get_frame_batch(self, l, r):
        frames = np.zeros((r-l, 1,) + self.frame_shape)
        cur_idx = 0
        while cur_idx+l < r:
            self.load_range(cur_idx+l)
            if self.frame_r > r:  # copy part of buffer
                frames[cur_idx:, 0, ...] = self.frame_buffer[:r-l-cur_idx, ...]
            else:  # copy everthing
                frames[cur_idx:self.frame_r-l, 0, ...] = self.frame_buffer[:, ...]
            cur_idx = self.frame_r-l
        return frames


    # load batch
    def load_range(self, l):
        self.frame_l = l
        self.frame_r = l + self.batch_size
        self.frame_buffer = np.zeros((self.frame_r-self.frame_l,) + self.frame_shape)
        for idx in range(self.frame_l, self.frame_r):
            self.frame_buffer[idx-self.frame_l, :] = self._load_image(idx)

    # load a scene
    def _load_image(self, idx):
        try:
            img = imio.imread(self._get_image_path(idx), as_gray=True) / 256.0
        except FileNotFoundError:
            img = np.full(self.frame_shape, 0)
        return img


    # image path
    def _get_image_path(self, idx):
        return os.path.join(self.state_path, 'state%d.png'%(idx))


def plot_focus(bo_game, all_focus):
    PX, PY, SHIFT = 2, 10, 20
    focus_img = bo_game.extract_neighbor(
        list(range(len(all_focus))),
        all_focus,
        nb_size=(15, 20)
    )
    for i in range(PY):
        plt.subplot(PX, PY, i + 1)
        plt.imshow(bo_game.get_frame(i+SHIFT))

        plt.subplot(PX, PY, PY + i + 1)
        plt.imshow(focus_img[i+SHIFT])
    plt.show()


# test loading 12 images with batch 5
if __name__ == '__main__':
    bo_game = BreakoutLog('selfBreakout/runs', 'selfBreakout/runs/0', batch_size=5)
    INIT = 321
    imgs =  bo_game.get_frame_batch(INIT, INIT+12)
    for i in range(12):
        plt.subplot(3, 4, i+1)
        # img = bo_game.get_frame(i+INIT)
        img = imgs[i]
        plt.imshow(img[0, ...])
    plt.show()