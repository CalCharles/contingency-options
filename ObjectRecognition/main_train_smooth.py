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
    ModelFocusCNN, ModelAttentionCNN,
    ModelCollectionDAG, load_param)
import ObjectRecognition.add_args as add_args
import ObjectRecognition.util as util


# get frame from dataset, mode in ['rand', 'range']
def get_frames(dataset, n_state, n_state_used, mode='rand'):
    assert mode in ['rand', 'range']
    if mode == 'rand':
        state_idxs = np.random.choice(np.arange(n_state), size=n_state_used, 
                                      replace=False)
    elif mode == 'range':
        state_idxs = np.arange(n_state_used)
    return torch.from_numpy(dataset.get_frame(state_idxs)).float()


if __name__ == '__main__':
    net_path = 'ObjectRecognition/net_params/attn_softmax.json'
    model_path = 'results/cmaes_soln/focus_atari_breakout/42531_2_smooth_4.pth'
    image_shape = (84, 84)
    n_state_used = 200
    is_train = True
    is_preview = False
    n_epoch = 50  # int or None
    n_iter = 10

    # get dataset
    binarize = 0.01
    n_state = 2000
    offset_fix = 0
    GAME_NAME = 'atari-ball'
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
    prev_weight_path_2 = 'results/cmaes_soln/focus_atari_breakout/42531_2_smooth_3_2.pth'
    prev_net_params_2 = json.loads(open(prev_net_params_path_2).read())
    prev_model_2 = ModelFocusCNN(
        image_shape=(84, 84),
        net_params=prev_net_params_2,
    )
    prev_model_2.set_parameters(load_param(prev_weight_path_2))
    prev_model = ModelCollectionDAG()
    prev_model.add_model('model_1', prev_model_1, [], 
                         augment_fn=partial(util.remove_mean_batch, nb_size=(3, 8)),
                         augment_pt=util.JumpFiltering(2, 0.05))
    prev_model.add_model('model_2', prev_model_2, ['model_1'],
                         augment_pt=util.JumpFiltering(3, 0.1))
    def prev_forward(xs, ret_extra=False):
        model_name = 'model_2'
        out = prev_model.forward(xs, ret_numpy=True, ret_extra=ret_extra)
        if ret_extra:
            return out[0][model_name], out[1][model_name]
        return out[model_name]
    print('smoothing:', prev_model)

    # get dataset
    if is_train and is_preview:
        frames = get_frames(dataset, n_state, n_state_used, mode='rand')
        focus_attn = util.focus2attn(
            prev_forward(frames),
            image_shape,
            fn=partial(util.gaussian_pdf, normalized=False))
        NC = 20
        for i in range(0, n_state_used, NC):
            fig, axes = plt.subplots(ncols=NC, nrows=2, figsize=(NC, 2))
            for x, y, ax in zip(frames[i:], focus_attn[i:], axes.T):
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
        for i in range(n_epoch):
            frames = get_frames(dataset, n_state, n_state_used, mode='rand')
            model.from_focus_model(
                prev_forward,
                frames,
                lr=1e-3 * (0.1**(i//20)),
                n_iter=n_iter,
                epoch=i,
            )

    else:
        model.load_state_dict(torch.load(model_path))
        model.eval()

    # forward
    frames = get_frames(dataset, n_state, n_state_used, mode='rand')
    focus_attn = util.focus2attn(
        prev_forward(frames),
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