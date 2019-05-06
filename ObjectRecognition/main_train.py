#!/usr/bin/env python

"""
Train a focus network

example:

python ObjectRecognition/main_train.py \
    results/cmaes_soln/focus_atari_breakout atari \
    ObjectRecognition/net_params/two_layer.json \
    --n_state 1000 \
    --binarize 0.1 \
    --popsize 20 \
    --saliency 0.2 1.0 0.02 \
    --action_micp 1.0 0.2 \
    --premise_micp 1.0 0.05 0.1 0.1 \
    --premise_path results/cmaes_soln/focus_self/paddle.npy \
    --premise_net ObjectRecognition/net_params/two_layer.json \
    --boost ObjectRecognition/net_params/two_layer.json \
        results/cmaes_soln/focus_atari_breakout/42080_16.npy \
    --verbose

"""

from __future__ import division, absolute_import, print_function
from functools import partial
import numpy as np
import json
import torch

import logging
logging.basicConfig(format='%(levelname)s [%(asctime)s]: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

from SelfBreakout.breakout_screen import RandomConsistentPolicy, RotatePolicy
from ChangepointDetection.LinearCPD import LinearCPD
from ChangepointDetection.CHAMP import CHAMPDetector
from ObjectRecognition.dataset import DatasetSelfBreakout, DatasetAtari
from ObjectRecognition.model import (
    ModelFocusCNN, ModelFocusBoost,
    ModelAttentionCNN,
    ModelCollectionDAG,
    load_param)
from ObjectRecognition.optimizer import CMAEvolutionStrategyWrapper
from ObjectRecognition.train import Trainer, recognition_train
from ObjectRecognition.loss import (
    SaliencyLoss, ActionMICPLoss, PremiseMICPLoss,
    CollectionMICPLoss, CombinedLoss,
    AttentionPremiseMICPLoss)
import ObjectRecognition.add_args as add_args
import ObjectRecognition.util as util

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train object recognition')
    parser.add_argument('savedir',
                        help='base directory to save results')
    parser.add_argument('game', choices=['self', 'self-b', 'atari'],
                        help='game name to train with')
    parser.add_argument('net',
                        help='network params JSON file')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='number of training iterations')
    add_args.add_changepoint_argument(parser)
    add_args.add_dataset_argument(parser)
    add_args.add_optimizer_argument(parser)
    add_args.add_model_argument(parser)
    add_args.add_loss_argument(parser)
    args = parser.parse_args()
    logger.info('arguments: %s', str(args))


    # CHAMP parameters
    if args.champ == 'ball':
        logger.info('using CHAMP ball parameters')
        CHAMP_params = [15, 10, 2, 100, 100, 2, 1, 0]
    elif args.champ == 'paddle':
        logger.info('using CHAMP paddle parameters')
        CHAMP_params = [3, 5, 1, 100, 100, 2, 1e-1, 0]

    # cheating flag
    if args.cheating:  # only 1 filter is allowed
        args.net = 'ObjectRecognition/net_params/one_filter.json'


    """
    Game Construction
        - SelfBreakout
        - Atari OpenAI Gym
    """
    if args.game == 'self':
        dataset = DatasetSelfBreakout(
            'SelfBreakout/runs',  # object dump path
            'SelfBreakout/runs/0',  # run states
            n_state=args.n_state,  # set max number of states
            binarize=args.binarize,  # binarize image to 0 and 1
            offset_fix=args.offset_fix,  # offset of episode number
        )  # 10.0, 0.1, 1.0, 0.0005
    elif args.game == 'self-b':
        dataset = DatasetSelfBreakout(
            'SelfBreakout/runs_bounce',  # object dump path
            'SelfBreakout/runs_bounce/0',  # run states
            n_state=args.n_state,  # set max number of states
            binarize=args.binarize,  # binarize image to 0 and 1
            offset_fix=args.offset_fix,  # offset of episode number
        )  # 10.0, 0.1, 1.0, 0.0005
    elif args.game == 'atari':
        # actor = partial(RandomConsistentPolicy, change_prob=0.35)
        actor = partial(RotatePolicy, hold_count=5)
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
        logger.info('using simple linear changepoint detector')
        cpd = LinearCPD(np.pi/4.0)


    """
    Model Template & Constructor
    """
    model = ModelCollectionDAG()
    net_params = json.loads(open(args.net).read())
    if args.model_type == 'focus':
        train_model = ModelFocusCNN(
            image_shape=dataset.frame_shape,
            net_params=net_params,
            use_prior=args.prior,
            argmax_mode=args.argmax_mode,
        )
    elif args.model_type == 'attn':
        train_model = ModelAttentionCNN(
            image_shape=dataset.frame_shape,
            net_params=net_params,
        )
    logger.info('loaded net_params %s'%(str(net_params)))

    # boosting with trained models
    if args.boost:
        # a model to be boosted
        b_net_params_path, b_weight_path = args.boost
        b_net_params = json.loads(open(b_net_params_path).read())
        b_params = load_param(b_weight_path)
        b_model = ModelFocusCNN(
            image_shape=(84, 84),
            net_params=b_net_params,
        )
        b_model.set_parameters(b_params)

        # boosting ensemble
        train_model = ModelFocusBoost(
            b_model,
            train_model,
            train_flags=[False, True],
            cp_detector=cpd,
        )

    # paddle model for premise MICP loss
    if args.premise_micp or args.attn_premise_micp:
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
        model.add_model('premise', pmodel, [], augment_fn=util.remove_mean_batch)
        # model.add_model('premise', pmodel, [],
        #                 augment_fn=util.RemoveMeanMemory(nb_size=(5, 5)))
        model.add_model('train', train_model, ['premise'])
    else:
        model.add_model('train', train_model, [])
    model.set_trainable('train')
    logger.info(model)


    """
    Loss Initiliazation
        - Focus losses
            - saliency
            - mutual information with action
            - OR MI with premise
    """
    seq_loss = []
    if args.saliency:
        saliencyloss = SaliencyLoss(
            dataset,
            c_fn_2=partial(util.hinged_mean_square_deviation, 
                           alpha_d=args.hinge_dist),  # 0.2
            frame_dev_coeff=args.saliency[0],  # 1.0
            focus_dev_coeff=args.saliency[1],  # 0.2
            frame_var_coeff=args.saliency[2],  # 0.02
            belief_dev_coeff=args.saliency[3],  # ???
            nb_size= (10, 10),  # TODO: auto-parameter?
            verbose=args.verbose,
        )
        seq_loss.append(saliencyloss)
    if args.action_micp or args.premise_micp:
        seq_micploss = []
        if args.action_micp:
            action_micploss = ActionMICPLoss(
                dataset,
                mi_match_coeff=args.action_micp[0],  # 1.0
                mi_diffs_coeff=args.action_micp[1],  # 0.2
                verbose=args.verbose,
            )
            seq_micploss.append(action_micploss)
        if args.premise_micp:
            premise_micploss = PremiseMICPLoss(
                dataset,
                'premise',
                mi_match_coeff=args.premise_micp[0],  # 1.0
                mi_diffs_coeff=args.premise_micp[1],  # 0.2
                mi_valid_coeff=args.premise_micp[2],  # 0.1
                mi_cndcp_coeff=args.premise_micp[3],  # 1.0
                prox_dist=args.premise_micp[4],  # 0.1
                batch_size=500,
                verbose=args.verbose,
            )
            seq_micploss.append(premise_micploss)
        micploss = CollectionMICPLoss(
            *seq_micploss,
            agg_fn=np.mean,
            cp_detector=cpd,
        )
        seq_loss.append(micploss)
    if args.attn_premise_micp:
        attn_premise_micploss = AttentionPremiseMICPLoss(
            dataset,
            'premise',
            mi_match_coeff=args.attn_premise_micp[0],  # ?
            mi_diffs_coeff=args.attn_premise_micp[1],  # ?
            temp_loss_coeff=args.attn_premise_micp[2],  # ?
            prox_dist=args.attn_premise_micp[3],  # ?
            attn_t=args.attn_premise_micp[4],  # ?
            batch_size=500,
            verbose=args.verbose,
        )
        seq_loss.append(attn_premise_micploss)
    loss = CombinedLoss(*seq_loss)
    logger.info(loss)


    """
    Optimizer
    """
    cmaes_opt = CMAEvolutionStrategyWrapper(
        model.count_parameters(),
        save_path=args.savedir,
        max_niter=args.niter,
        nproc=args.nproc,
        cheating=args.cheating,
        cmaes_params={
            'popsize': args.popsize,
        },
    )


    """
    Execution
    """
    result = recognition_train(
        dataset, 
        model, 
        loss.forward, 
        cmaes_opt, 
        verbose=args.verbose,
    )
    logger.info("result params: %s", str(result))
    Trainer(
        dataset, 
        model, 
        loss.forward,
        verbose=args.verbose,
    ).evaluate_model(result)