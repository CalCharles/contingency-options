import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from functools import partial

from SelfBreakout.breakout_screen import RandomConsistentPolicy, RotatePolicy
from ObjectRecognition.dataset import parse_dataset
from ObjectRecognition.model import (
    ModelFocusCNN, ModelFocusMeanVar,
    ModelCollectionDAG, load_param)
import ObjectRecognition.add_args as add_args
import ObjectRecognition.util as util


if __name__ == '__main__':
    model_path = 'results/cmaes_soln/focus_atari_breakout/mv_test.json'
    image_shape = (84, 84)
    n_state_used = 20

    # get dataset
    n_state = 500
    offset_fix = 0
    binarize = 0.01
    GAME_NAME = 'atari'
    dataset = parse_dataset(
        dataset_name=GAME_NAME,
        n_state=n_state,
        binarize=binarize,
        offset_fix=offset_fix
    )

    # get ball model
    prev_net_params_path_1 = 'ObjectRecognition/net_params/attn_softmax.json'
    prev_weight_path_1 = 'results/cmaes_soln/focus_atari_breakout/paddle_bin_smooth.pth'
    prev_net_params_1 = json.loads(open(prev_net_params_path_1).read())
    prev_model_1 = ModelFocusCNN(
        image_shape=(84, 84),
        net_params=prev_net_params_1,
    )
    prev_model_1.set_parameters(load_param(prev_weight_path_1))
    prev_net_params_path_2 = 'ObjectRecognition/net_params/attn_softmax.json'
    prev_weight_path_2 = 'results/cmaes_soln/focus_atari_breakout/42531_2_smooth_2.pth'
    prev_net_params_2 = json.loads(open(prev_net_params_path_2).read())
    prev_model_2 = ModelFocusCNN(
        image_shape=(84, 84),
        net_params=prev_net_params_2,
    )
    prev_model_2.set_parameters(load_param(prev_weight_path_2))
    prev_model = ModelCollectionDAG()
    prev_model.add_model('model_1', prev_model_1, [], 
                         augment_fn=partial(util.remove_mean_batch, nb_size=(3, 8)))
    # prev_model.add_model('model_2', prev_model_2, ['model_1'])
    f1 = util.LowIntensityFiltering(5.0)
    f2 = util.JumpFiltering(3, 0.05)
    def f(x, y):
        return f2(x, f1(x, y))
    prev_model.add_model('model_2', prev_model_2, ['model_1'], augment_pt=f)
    def prev_forward(xs):
        outs = prev_model.forward(xs, ret_numpy=True, ret_extra=True)
        return outs[0]['model_2'], outs[1]['model_2']
    print('mean-var:', prev_model)

    # get dataset
    state_idxs = np.random.choice(np.arange(n_state//4), size=n_state_used, 
                                  replace=False)
    print(state_idxs)
    frames = dataset.get_frame(state_idxs)
    frames = torch.from_numpy(frames).float()
    focus = prev_forward(frames)

    # create the model
    model = ModelFocusMeanVar.from_focus_model(prev_forward, dataset, 
                                               nb_size=(10, 10), batch_size=1000)
    print(model)

    # plot stuffs
    fig, axes = plt.subplots(ncols=2, figsize=(3, 1.5))
    for ax, im in zip(axes, (model.img_mean[0], model.img_var[0])):
        im = (im - np.max(im)) / (np.max(im) - np.min(im))
        ax.imshow(im)
        ax.axis('off')
    plt.show()

    # forward
    focus_x, attn = model.forward(frames, ret_numpy=True, ret_extra=True)

    # plots
    fig, axes = plt.subplots(ncols=20, nrows=3, figsize=(20, 3))
    for x, z, f, ax in zip(frames, attn, focus_x, axes.T):
        xn = x[0].detach().numpy()
        ax[0].imshow(xn)
        ax[1].imshow(z)
        combine = (xn - np.mean(xn)) / np.std(xn)
        combine += (z - np.mean(z)) / np.std(z)
        marker_pos = np.around(f * dataset.get_shape()[2:]).astype(int)
        combine[marker_pos[0], :] = 1
        combine[:, marker_pos[1]] = 1
        ax[2].imshow(combine)
        print(np.max(xn[0]), np.min(xn[0]), np.max(z[0]), np.min(z[0]))
    for ax in axes.flatten():
        ax.axis('off')
    plt.show()