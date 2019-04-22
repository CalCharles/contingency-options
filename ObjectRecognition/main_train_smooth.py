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
    model_path = 'results/cmaes_soln/focus_self/42309_7_smooth_2.pth'
    # model_path = 'results/cmaes_soln/focus_atari_breakout/42101_22_smooth.pth'
    image_shape = (84, 84)
    n_state = 30
    is_train = False

    # get dataset
    dataset = DatasetSelfBreakout(
        'SelfBreakout/runs',  # object dump path
        'SelfBreakout/runs/0',  # run states2
        n_state=n_state,  # set max number of states
        binarize=0.1,  # binarize image to 0 and 1
        offset_fix=0,  # offset of episode number
    )
    # actor = partial(RotatePolicy, hold_count=5)
    # dataset = DatasetAtari(
    #     'BreakoutNoFrameskip-v4',  # atari game name
    #     actor,  # mock actor
    #     n_state=n_state,  # set max number of states
    #     save_path='results',  # save path for gym
    #     binarize=0.1,  # binarize image to 0 and 1
    # )

    # get ball model
    prev_net_params_path = 'ObjectRecognition/net_params/two_layer_5_5_old.json'
    prev_weight_path = 'results/cmaes_soln/focus_self/42309_7.npy'
    # prev_net_params_path = 'ObjectRecognition/net_params/two_layer.json'
    # prev_weight_path = 'results/cmaes_soln/focus_atari_breakout/42101_22.npy'
    prev_net_params = json.loads(open(prev_net_params_path).read())
    prev_model = ModelFocusCNN(
        image_shape=(84, 84),
        net_params=prev_net_params,
    )
    prev_model.set_parameters(load_param(prev_weight_path))

    # create the model
    frames = dataset.get_frame(0, n_state)
    frames = torch.from_numpy(frames).float()
    net_params = json.loads(open(net_path).read())
    model = ModelAttentionCNN(image_shape, net_params)
    print(model)

    # train
    if is_train:
        model.from_focus_model(prev_model, dataset, n_iter=300)
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