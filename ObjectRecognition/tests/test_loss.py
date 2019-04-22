from __future__ import division, absolute_import, print_function
from functools import partial
import json
import torch

import matplotlib.pyplot as plt
from ObjectRecognition.main_report import plot_focus
from SelfBreakout.breakout_screen import RandomConsistentPolicy
from ObjectRecognition.dataset import DatasetSelfBreakout, DatasetAtari
from ObjectRecognition.model import (
    ModelFocusCNN, ModelCollectionDAG,
    load_param)
from ObjectRecognition.loss import *
from ObjectRecognition.util import extract_neighbor


def load_model(model_path, net_params_path, pmodel=None, *args, **kwargs):
    net_params = json.loads(open(net_params_path).read())
    params = load_param(model_path)
    test_mode = ModelFocusCNN(
        image_shape=(84, 84),
        net_params=net_params,
        *args,
        **kwargs,
    )
    test_mode.set_parameters(params)

    # construct model
    model = ModelCollectionDAG()
    if pmodel:
        model.add_model('premise', pmodel, [])
        model.add_model('test_model', test_mode, ['premise'])
    else:
        model.add_model('test_model', test_mode, [])
    model.set_trainable('test_model')
    return model

# game instance
n_state = 1000
GAME_NAME = 'self-b'  # 'self', 'self-b', 'atari'
if GAME_NAME == 'self':
    game = DatasetSelfBreakout(
        'SelfBreakout/runs',
        'SelfBreakout/runs/0',
        binarize=0.1,
        n_state=n_state,
        offset_fix=0,
    )
elif GAME_NAME == 'self-b':
    game = DatasetSelfBreakout(
        'SelfBreakout/runs_bounce',
        'SelfBreakout/runs_bounce/0',
        binarize=0.1,
        n_state=n_state,
        offset_fix=0,
    )
elif GAME_NAME == 'atari':
    game = DatasetAtari(
        'BreakoutNoFrameskip-v4',
        partial(RandomConsistentPolicy, change_prob=0.35),
        n_state=1000,
        save_path='results',
        normalized_coor=True,
        offset_fix=0,
    )

# saliency
dmiloss = SaliencyLoss(
    game,
    c_fn_2=partial(util.hinged_mean_square_deviation, 
                   alpha_d=0.3),  # TODO: parameterize this
    frame_dev_coeff= 0.0,
    focus_dev_coeff= 20.0,
    frame_var_coeff= 0.0,
    belief_dev_coeff= 0.0,
    nb_size= (10, 10),
    verbose=True,
)

# action micp
micplosses = []
action_micploss = ActionMICPLoss(
    game,
    mi_match_coeff= 1.0,
    mi_diffs_coeff= 0.2,
    verbose=True,
)
micplosses.append(action_micploss)

# premise loss
pmodel_net_params_path = 'ObjectRecognition/net_params/two_layer.json'
net_params = json.loads(open(pmodel_net_params_path).read())
params = load_param('results/cmaes_soln/focus_self/paddle_bin.npy')
pmodel = ModelFocusCNN(
    image_shape=(84, 84),
    net_params=net_params,
)
pmodel.set_parameters(params)
paddle_model = load_model(
    'results/cmaes_soln/focus_self/paddle_bin.npy',
    'ObjectRecognition/net_params/two_layer.json',
    pmodel=pmodel)
ball_model = load_model(
    'results/cmaes_soln/focus_self/ball_bin.npy',
    'ObjectRecognition/net_params/two_layer.json',
    pmodel=pmodel)
comp_model = load_model(
    'results/cmaes_soln/focus_self/42068_40.npy',
    'ObjectRecognition/net_params/two_layer.json',
    pmodel=pmodel)

premise_micploss = PremiseMICPLoss(
    game,
    'premise',
    mi_match_coeff= 0.0,
    mi_diffs_coeff= 0.0,
    mi_valid_coeff= 0.0,
    mi_cndcp_coeff= 1.0,
    prox_dist= 0.1,
    verbose=True,
)
# micplosses.append(premise_micploss)

# combine every loss together
micploss = CollectionMICPLoss(*micplosses)
loss = CombinedLoss(dmiloss, micploss)
print(loss)
loss_fn = loss.forward

# get focus lists for comparison
LIMIT = n_state
L = game.idx_offset
R = L + LIMIT
paddle_model_focus = paddle_model.forward(torch.from_numpy(game.get_frame(0, LIMIT)).float(), ret_numpy=True)
ball_model_focus = ball_model.forward(torch.from_numpy(game.get_frame(0, LIMIT)).float(), ret_numpy=True)
comp_model_focus = comp_model.forward(torch.from_numpy(game.get_frame(0, LIMIT)).float(), ret_numpy=True)
random_focus = np.random.rand(LIMIT, 2)
random_focus = {
    'premise': paddle_model_focus['premise'],
    '__train__': random_focus,
}
try:
    ideal_paddle = game.paddle_data.astype(float)[L:R, ...] / 84.0
except:
    ideal_paddle = np.random.rand(LIMIT, 2)
ideal_paddle = {
    'premise': paddle_model_focus['premise'],
    '__train__': ideal_paddle,
}
try:
    ideal_ball = game.ball_data.astype(float)[L:R, ...] / 84.0
except:
    ideal_ball = np.random.rand(LIMIT, 2)
ideal_ball = {
    'premise': paddle_model_focus['premise'],
    '__train__': ideal_ball,
}
fix_jump_focus = paddle_model_focus
fix_jump_focus = np.zeros(fix_jump_focus['__train__'].shape)
fix_jump_focus['__train__'][[LIMIT//6, LIMIT//4, LIMIT//2]] = paddle_model_focus['__train__'][[LIMIT//6, LIMIT//4, LIMIT//2]]

# plot tracking paddle
if False:
    L, R = 20, 30
    plot_focus(game, range(L, R), ideal_paddle['__train__'][L:R])
    plot_focus(game, range(L, R), ideal_ball['__train__'][L:R])
    plot_focus(game, range(L, R), random_focus['__train__'][L:R])
    plot_focus(game, range(L, R), paddle_model_focus['__train__'][L:R])
    plot_focus(game, range(L, R), ball_model_focus['__train__'][L:R])
    plot_focus(game, range(L, R), comp_model_focus['__train__'][L:R])

# calculate loss
print('Ideal Paddle Loss:', loss_fn(ideal_paddle))
print('Ideal Ball Loss:', loss_fn(ideal_ball))
print('Random Loss:', loss_fn(random_focus))
print('Fix Jump Loss:', loss_fn(fix_jump_focus))
print('Model Paddle Loss:', loss_fn(paddle_model_focus))
print('Model Ball Loss:', loss_fn(ball_model_focus))
print('Model Compared Loss:', loss_fn(comp_model_focus))