#!/usr/bin/env python

"""
Example:
    python ObjectRecognition/main_report.py results/cmaes_soln/focus_atari_breakout atari ObjectRecognition/net_params/two_layer.json 42365_5 --plot-focus --plot-filter --plot-cp

"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cma
from functools import partial
from scipy.misc import imsave

from arguments import get_args

from SelfBreakout.breakout_screen import (
    read_obj_dumps, get_individual_data, hot_actions,
    RandomConsistentPolicy, RotatePolicy)
from ChangepointDetection.LinearCPD import LinearCPD
from ChangepointDetection.CHAMP import CHAMPDetector
from ObjectRecognition.dataset import DatasetSelfBreakout, DatasetAtari
from ObjectRecognition.model import ModelFocusCNN, ModelFocusBoost
from ObjectRecognition.optimizer import CMAEvolutionStrategyWrapper
from ObjectRecognition.loss import (
    SaliencyLoss, ActionMICPLoss, PremiseMICPLoss,
    CollectionMICPLoss, CombinedLoss)
import ObjectRecognition.util as util


def get_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('created directory', path)
    return path


def plot_model_filter(model):
    filters = [submodule[1].detach().numpy()
               for submodule in model.named_parameters() 
               if 'weight' in submodule[0] and len(submodule[1].shape) == 4]
    filters = [f.reshape((-1,) + f.shape[-2:]) for f in filters]
    NPLOT = sum(map(lambda f: f.shape[0], filters))
    N = int(np.ceil(NPLOT**0.5))
    M = NPLOT // N
    fig, axs = plt.subplots(nrows=M, ncols=N)
    axs_list = np.array(axs).reshape(-1)
    pidx = 0
    for f in filters:
        for subf in f:
            axs_list[pidx].imshow(subf)
            pidx += 1
    plt.show()


def load_model(prefix, model_id, net_params, cp_detector, use_prior=False):
    model_param_path = os.path.join(get_dir(prefix), '%s.npy'%model_id)
    params = np.load(model_param_path)
    model = ModelFocusCNN(
        image_shape=(84, 84),
        net_params=net_params,
        cp_detector=cp_detector,
        use_prior=use_prior,
    )
    model.set_parameters(params)

    return model


def plot_focus(dataset, indices, all_focus):
    PX, PY = 2, len(all_focus)
    focus_img = util.extract_neighbor(
        dataset,
        all_focus,
        indices,
        nb_size=(15, 20)
    )
    for i in range(PY):
        plt.subplot(PX, PY, i + 1)
        plt.imshow(dataset.get_frame(indices[i])[0])

        plt.subplot(PX, PY, PY + i + 1)
        plt.imshow(focus_img[i])
    plt.show()


def save_focus_img(dataset, all_focus, save_path, changepoints=None):
    focus_img = util.extract_neighbor(
        dataset,
        all_focus,
        range(len(all_focus)),
        nb_size=(15, 20)
    )
    for i, img in enumerate(focus_img):
        # focused neighbor image
        file_name = 'focus_img_%d.png'%(i)
        file_path = os.path.join(save_path, file_name)
        imsave(file_path, img)
    print('saved under', save_path)

    cp_mask = np.zeros(len(all_focus), dtype=bool)
    cp_mask[changepoints] = True
    save_subpath = get_dir(os.path.join(save_path, 'marker'))
    for i, img in enumerate(dataset.get_frame(0, len(all_focus))):
        # marker
        file_name = 'marker_%d.png'%(i)
        marker_file_path = os.path.join(save_subpath, file_name)
        marker_pos = (all_focus[i] * dataset.get_shape()[2:]).astype(int)
        img[0, marker_pos[0], :] = 1
        img[0, :, marker_pos[1]] = 1
        if cp_mask[i]:
            img = 1 - img
        imsave(marker_file_path, img[0])
    print('focus by marker saved under', save_subpath)



def report_model(dataset, model, prefix, plot_flags, cpd):
    focus_img_dir = get_dir(os.path.join(prefix, 'focus_img_%s'%model_id))
    focus = model.forward_all(dataset, batch_size=400)
    if plot_flags['plot_focus_live']:
        N, PY = 1000, 10
        for i in range(0, N-PY, PY):
            plot_focus(dataset, range(i, i+PY), focus[i:i+PY])
    if plot_flags['plot_focus']:
        # compute changepoints if needed
        changepoints = None
        if plot_flags['plot_cp']:
            changepoints = cpd.generate_changepoints(focus)
            # _, changepoints = CHAMP.generate_changepoints(
            #     [LinearDynamicalDisplacementFitter],
            #     CHAMP.CHAMP_parameters(15, 10, 2, 100, 100, 2, 1),  # ball
            #     focus)

        # plot and save into directories
        save_focus_img(dataset, focus, focus_img_dir, changepoints)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Report for Object Recognition')
    parser.add_argument('dir',
                        help='base directory')
    parser.add_argument('game', choices=['self', 'atari'],
                        help='game name to report with')
    parser.add_argument('net',
                        help='network params JSON file')
    parser.add_argument('modelID',
                        help='model params file')
    parser.add_argument('--champ', choices=['ball', 'paddle'],
                        help='parameters for CHAMP')
    parser.add_argument('--boost', action='store_true', default=False,
                        help='boost mode [in progress]')
    parser.add_argument('--prior', action='store_true', default=False,
                        help='use focus prior filter')
    parser.add_argument('--plot-filter', action='store_true', default=False,
                        help='plot model filter')
    parser.add_argument('--plot-focus', action='store_true', default=False,
                        help='plot focus neighborhood and save to directory')
    parser.add_argument('--plot-focus-live', action='store_true', default=False,
                        help='plot focus neighborhood every timestep')
    parser.add_argument('--plot-cp', action='store_true', default=False,
                        help='indicate changepoint frames (with --plot-focus)')
    args = parser.parse_args()

    prefix = args.dir
    model_id = args.modelID
    plot_flags = {
        'plot_filter': args.plot_filter,
        'plot_focus': args.plot_focus,
        'plot_focus_live': args.plot_focus_live,
        'plot_cp': args.plot_cp,
    }

    # CHAMP parameters
    if args.champ == 'ball':
        print('using CHAMP ball parameters')
        CHAMP_params = CHAMP_BALL_PARAMETERS
    elif args.champ == 'paddle':
        print('using CHAMP paddle parameters')
        CHAMP_params = CHAMP_PADDLE_PARAMETERS

    """
    Game Construction
    """
    # pick game
    if args.game == 'self':
        dataset = DatasetSelfBreakout(
            'SelfBreakout/runs',  # object dump path
            'SelfBreakout/runs/0',  # run states
        )  # 10.0, 0.1, 1.0, 0.0005
    elif args.game == 'atari':
        actor = partial(RotatePolicy, hold_count=8)
        dataset = DatasetAtari(
            'BreakoutNoFrameskip-v4',  # atari game name
            actor,  # mock actor
            n_state=1000,  # set max number of states
            save_path='results',  # save path for gym
        )

    """
    Changepoint Detector
        - specify changepoint detector which fits the dynamic of the object
    """
    if args.champ:
        cpd = CHAMPDetector([LinearDynamicalDisplacementFitter], CHAMP_params)
    else:
        print('using simple linear changepoint detector')
        cpd = LinearCPD(np.pi/4.0)

    """
    Load model
    """
    net_params = json.loads(open(args.net).read())
    model = load_model(
        prefix,
        model_id,
        net_params,
        cpd,
        use_prior=args.prior,
    )
    if plot_flags['plot_filter']:
        plot_model_filter(model)

    # boosting with trained models
    if args.boost:
        # partial ball model to be boosted
        ball_model_id = 'results/cmaes_soln/focus_atari_breakout/42080_16.npy'
        ball_net_params_text = open('objRecog/net_params/two_layer.json').read()
        ball_net_params = json.loads(ball_net_params_text)
        ball_params = np.load(ball_model_id)
        ball_model = ModelFocusCNN(
            image_shape=(84, 84),
            net_params=ball_net_params,
        )
        ball_model.set_parameters(ball_params)

        # boosting ensemble
        model = ModelFocusBoost(
            ball_model,
            model,
            train_flags=[False, True],
            cp_detector=cpd,
        )

    """
    Do the report
    """
    report_model(dataset, model, prefix, plot_flags, cpd)

    # prefix = 'results/cmaes_soln/transfer'
    # for i in range(39, 559+1, 40):
    #     model_id = '42394_%d'%i
    #     report_model(prefix, model_id)
