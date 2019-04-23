import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from functools import partial

from SelfBreakout.breakout_screen import RandomConsistentPolicy, RotatePolicy
from ObjectRecognition.dataset import DatasetSelfBreakout, DatasetAtari
from ObjectRecognition.model import ModelFocusCNN, ModelAttentionCNN, load_param
import ObjectRecognition.add_args as add_args
import ObjectRecognition.util as util


if __name__ == '__main__':
    net_path = 'ObjectRecognition/net_params/attn_base.json'
    model_path = 'results/cmaes_soln/focus_self/paddle_bin_long_smooth_2.pth'
    # model_path = 'results/cmaes_soln/focus_atari_breakout/42101_22_smooth.pth'
    image_shape = (84, 84)
    n_state_used = 100
    is_train = False

    # get dataset
    n_state = 10000
    offset_fix = 0
    GAME_NAME = 'self-b'  # 'self', 'self-b', 'atari'
    if GAME_NAME == 'self':
        dataset = DatasetSelfBreakout(
            'SelfBreakout/runs',
            'SelfBreakout/runs/0',
            binarize=0.1,
            n_state=n_state,
            offset_fix=offset_fix,
        )
    elif GAME_NAME == 'self-b':
        dataset = DatasetSelfBreakout(
            'SelfBreakout/runs_bounce',
            'SelfBreakout/runs_bounce/0',
            binarize=0.1,
            n_state=n_state,
            offset_fix=offset_fix,
        )
    elif GAME_NAME == 'atari':
        dataset = DatasetAtari(
            'BreakoutNoFrameskip-v4',
            partial(RandomConsistentPolicy, change_prob=0.35),
            n_state=n_state,
            save_path='results',
            normalized_coor=True,
            offset_fix=offset_fix,
        )

    # get ball model
    prev_net_params_path = 'ObjectRecognition/net_params/attn_base.json'
    prev_weight_path = 'results/cmaes_soln/focus_self/paddle_bin_long_smooth.pth'
    # prev_net_params_path = 'ObjectRecognition/net_params/two_layer.json'
    # prev_weight_path = 'results/cmaes_soln/focus_atari_breakout/42101_22.npy'
    prev_net_params = json.loads(open(prev_net_params_path).read())
    prev_model = ModelFocusCNN(
        image_shape=(84, 84),
        net_params=prev_net_params,
    )
    prev_model.set_parameters(load_param(prev_weight_path))

    # get dataset
    state_idxs = np.random.choice(np.arange(n_state), size=n_state_used, 
                                  replace=False)
    print(state_idxs) 
    frames = dataset.get_frame(state_idxs)
    frames = torch.from_numpy(frames).float()
    focus = prev_model.forward(frames)
    focus_attn = util.focus2attn(
        focus,
        image_shape,
        fn=partial(util.gaussian_pdf, normalized=False))
    if is_train:
        NC = 20
        for i in range(0, n_state_used, NC):
            fig, axes = plt.subplots(ncols=NC, nrows=2, figsize=(NC, 2))
            for x, y, ax in zip(frames, focus_attn[i:], axes.T):
                xn = x[0].detach().numpy()
                ax[0].imshow(xn + y[0])
                ax[1].imshow(y[0])
            plt.show()

    # create the model
    net_params = json.loads(open(net_path).read())
    model = ModelAttentionCNN(image_shape, net_params)
    print(model)


    # train
    if is_train:
        # model.from_focus_model(prev_model, dataset, n_iter=300)
        model.from_focus_model(prev_model, frames, n_iter=300)
    else:
        model.load_state_dict(torch.load(model_path))
        model.eval()

    # forward
    focus = prev_model.forward(frames)
    focus_attn = util.focus2attn(
        focus,
        image_shape,
        fn=partial(util.gaussian_pdf, normalized=False))
    attn = model.forward(frames, ret_numpy=True)

    # plots
    fig, axes = plt.subplots(ncols=20, nrows=4, figsize=(20, 8))
    for x, y, z, ax in zip(frames, focus_attn, attn, axes.T):
        xn = x[0].detach().numpy()
        ax[0].imshow(xn + y[0])
        ax[1].imshow(y[0])
        ax[2].imshow(z[0])
        ax[3].imshow(xn + z[0])
        print(np.max(xn[0]), np.min(xn[0]), np.max(z[0]), np.min(z[0]))
    plt.show()

    if is_train:
        # save
        torch.save(model.state_dict(), model_path)

        # test loading
        model_2 = ModelAttentionCNN(image_shape, net_params)
        model_2.load_state_dict(torch.load(model_path))
        model_2.eval()
        attn_2 = model.forward(frames, ret_numpy=True)
        fig, axes = plt.subplots(ncols=10, nrows=2, figsize=(20, 2))
        for x, y, ax in zip(attn, attn_2, axes.T):
            ax[0].imshow(x[0])
            ax[1].imshow(y[0])
        plt.show()