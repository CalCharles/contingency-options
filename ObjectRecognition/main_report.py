#!/usr/bin/env python

"""
Example:
    python ObjectRecognition/main_report.py results/cmaes_soln/focus_atari_breakout atari ObjectRecognition/net_params/two_layer.json 42365_5 --plot-focus --plot-filter --plot-cp

"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import cma
from functools import partial

from arguments import get_args

from SelfBreakout.breakout_screen import (
    read_obj_dumps, get_individual_data, hot_actions,
    RandomConsistentPolicy, RotatePolicy)
from ChangepointDetection.LinearCPD import LinearCPD
from ChangepointDetection.CHAMP import CHAMPDetector
from ObjectRecognition.dataset import DatasetSelfBreakout, DatasetAtari
from ObjectRecognition.model import (
    ModelFocusCNN, ModelFocusBoost, 
    ModelCollectionDAG, load_param)
from ObjectRecognition.optimizer import CMAEvolutionStrategyWrapper
from ObjectRecognition.loss import (
    SaliencyLoss, ActionMICPLoss, PremiseMICPLoss,
    CollectionMICPLoss, CombinedLoss)
import ObjectRecognition.add_args as add_args
import ObjectRecognition.util as util


def plot_model_filter(model, save_path=None):
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
            axs_list[pidx].imshow(subf, interpolation=None)
            pidx += 1

    if save_path is not None:
        file_path = os.path.join(save_path, 'filter.png')
        plt.savefig(file_path)
        print('saved weight at', save_path)
    else:
        plt.show()
    plt.close()


def load_model(prefix, model_id, net_params, *args, **kwargs):
    params = load_param((util.get_dir(prefix), model_id))
    model = ModelFocusCNN(
        image_shape=(84, 84),
        net_params=net_params,
        *args,
        **kwargs,
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
        plt.imshow(dataset.get_frame(indices[i])[0], interpolation=None)

        plt.subplot(PX, PY, PY + i + 1)
        plt.imshow(focus_img[i], interpolation=None)
    plt.show()


def save_imgs(imgs, save_path):
    save_subpath = util.get_dir(os.path.join(save_path, 'intensity'))
    for i, img in enumerate(imgs):
        file_name = 'intensity_%d.png'%(i)
        file_path = os.path.join(save_subpath, file_name)
        util.imsave(file_path, util.feature_normalize(img[0]))
    print('focus intensity saved under', save_subpath)


def save_focus_img(dataset, all_focus, save_path, changepoints=[]):
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
        util.imsave(file_path, img)
    print('saved under', save_path)

    pos_cnt = np.zeros(dataset.get_shape()[-2:])
    cp_mask = np.zeros(len(all_focus), dtype=bool)
    cp_mask[changepoints] = True
    save_subpath = util.get_dir(os.path.join(save_path, 'marker'))
    for i, img in enumerate(dataset.get_frame(0, len(all_focus))):
    # for i, img in enumerate(util.remove_mean(dataset.get_frame(0, len(all_focus)), all_focus)):
        # marker
        file_name = 'marker_%d.png'%(i)
        marker_file_path = os.path.join(save_subpath, file_name)
        marker_pos = np.around(all_focus[i] * dataset.get_shape()[2:]).astype(int)
        img[0, marker_pos[0], :] = 1
        img[0, :, marker_pos[1]] = 1
        if cp_mask[i]:
            img = 1 - img
        pos_cnt[marker_pos[0], marker_pos[1]] += 1
        util.imsave(marker_file_path, img[0])
    print('focus by marker saved under', save_subpath)

    if np.sum(pos_cnt) > 0:
        pos_cnt_file_path = os.path.join(save_path, 'counter_pos.png')
        util.count_imsave(pos_cnt_file_path, pos_cnt)
        print('position count saved at', pos_cnt_file_path)


def report_model(save_path, dataset, model, prefix, plot_flags, cpd):
    focus = model.forward_all(dataset, batch_size=400, 
                              ret_extra=plot_flags['plot_intensity'])

    if plot_flags['plot_intensity']:
        save_imgs(focus[1]['__train__'], save_path)
        focus = focus[0]
    else:
        focus = focus

    if plot_flags['plot_loss']:
        print('loss eval:', plot_flags['plot_loss'].forward(focus))

    if plot_flags['plot_focus']:
        # compute changepoints if needed
        changepoints = []
        if plot_flags['plot_cp']:
            _, changepoints = cpd.generate_changepoints(focus['__train__'])

        # plot and save into directories
        save_focus_img(dataset, focus['__train__'], save_path, changepoints)

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
    parser.add_argument('--plot-filter', action='store_true', default=False,
                        help='plot model filter')
    parser.add_argument('--plot-focus', action='store_true', default=False,
                        help='plot focus neighborhood and save to directory')
    parser.add_argument('--plot-cp', action='store_true', default=False,
                        help='indicate changepoint frames (with --plot-focus)')
    parser.add_argument('--plot-intensity', action='store_true', default=False,
                        help='plot focus intensity (output before argmax)')
    parser.add_argument('--plot-loss', action='store_true', default=False,
                        help='evaluate and plot loss function')
    add_args.add_changepoint_argument(parser)
    add_args.add_dataset_argument(parser)
    add_args.add_model_argument(parser)
    args = parser.parse_args()
    print(args)

    prefix = args.dir
    model_id = args.modelID
    plot_flags = {
        'plot_filter': args.plot_filter,
        'plot_focus': args.plot_focus,
        'plot_cp': args.plot_cp,
        'plot_intensity': args.plot_intensity,
        'plot_loss': args.plot_loss,
    }

    # CHAMP parameters
    if args.champ == 'ball':
        logger.info('using CHAMP ball parameters')
        CHAMP_params = [15, 10, 2, 100, 100, 2, 1, 0]
    elif args.champ == 'paddle':
        logger.info('using CHAMP paddle parameters')
        CHAMP_params = [3, 5, 1, 100, 100, 2, 1e-1, 0]

    """
    Game Construction
    """
    # pick game
    if args.game == 'self':
        dataset = DatasetSelfBreakout(
            'SelfBreakout/runs_bounce',  # object dump path
            'SelfBreakout/runs_bounce/0',  # run states
            n_state=args.n_state,  # set max number of states
            binarize=args.binarize,  # binarize image to 0 and 1
            offset_fix=args.offset_fix,  # offset of episode number
        )  # 10.0, 0.1, 1.0, 0.0005
    elif args.game == 'atari':
        actor = partial(RotatePolicy, hold_count=4)
        dataset = DatasetAtari(
            'BreakoutNoFrameskip-v4',  # atari game name
            actor,  # mock actor
            n_state=args.n_state,  # set max number of states
            save_path='results',  # save path for gym
            binarize=args.binarize,  # binarize image to 0 and 1
        )

    """
    Changepoint Detector
        - specify changepoint detector which fits the dynamic of the object
    """
    if args.champ:
        cpd = CHAMPDetector('premise->object', CHAMP_params)
    else:
        print('using simple linear changepoint detector')
        cpd = LinearCPD(np.pi/4.0)

    """
    Load model
    """
    net_params = json.loads(open(args.net).read())
    r_model = load_model(
        prefix,
        model_id,
        net_params=net_params,
        use_prior=args.prior,
        argmax_mode=args.argmax_mode,
    )
    save_path = util.get_dir(os.path.join(prefix, 'focus_img_%s'%model_id))
    if plot_flags['plot_filter']:
        plot_model_filter(r_model, save_path)

    # boosting with trained models
    if args.boost:
        # partial ball model to be boosted
        ball_model_id = 'results/cmaes_soln/focus_atari_breakout/42080_16.npy'
        ball_net_params_text = open('objRecog/net_params/two_layer.json').read()
        ball_net_params = json.loads(ball_net_params_text)
        ball_params = load_param(ball_model_id)
        ball_model = ModelFocusCNN(
            image_shape=(84, 84),
            net_params=ball_net_params,
        )
        ball_model.set_parameters(ball_params)

        # boosting ensemble
        r_model = ModelFocusBoost(
            ball_model,
            r_model,
            train_flags=[False, True],
            cp_detector=cpd,
        )
    model = ModelCollectionDAG()
    if args.premise_path:
        pmodel_weight_path = args.premise_path
        pmodel_net_params_text = open(args.premise_net).read()
        pmodel_net_params = json.loads(pmodel_net_params_text)
        pmodel_params = load_param(pmodel_weight_path)
        pmodel = ModelFocusCNN(
            image_shape=(84, 84),
            net_params=pmodel_net_params,
        )
        pmodel.set_parameters(pmodel_params)
        # model.add_model('premise', pmodel, [])
        # model.add_model('premise', pmodel, [], augment_fn=util.remove_mean_batch)
        model.add_model('premise', pmodel, [], 
                         augment_fn=partial(util.remove_mean_batch, nb_size=(8, 8)))
        # model.add_model('premise', pmodel, [], 
        #                 augment_fn=util.RemoveMeanMemory(nb_size=(5, 5)))
        model.add_model('train', r_model, ['premise'])
    else:
        model.add_model('train', r_model, [])
    model.set_trainable('train')
    print(model)

    """
    Load a loss
    """
    if plot_flags['plot_loss']:
        premise_micploss = PremiseMICPLoss(
            dataset,
            'premise',
            mi_match_coeff= 0.0,
            mi_diffs_coeff= 0.0,
            mi_valid_coeff= 0.0,
            mi_cndcp_coeff= 1.0,
            prox_dist= 0.090,
            verbose=True,
        )
        micploss = CollectionMICPLoss(premise_micploss)
        loss = CombinedLoss([micploss])
        print(loss)
        plot_flags['plot_loss'] = loss

    """
    Do the report
    """
    report_model(save_path, dataset, model, prefix, plot_flags, cpd)