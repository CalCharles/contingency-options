from __future__ import division, absolute_import, print_function
from functools import partial
import json
import torch

import matplotlib.pyplot as plt
from ObjectRecognition.main_report import plot_focus
from SelfBreakout.breakout_screen import RandomConsistentPolicy
from ObjectRecognition.dataset import DatasetSelfBreakout, DatasetAtari
from ObjectRecognition.model import ModelFocusCNN
from ObjectRecognition.loss import *
from ObjectRecognition.util import extract_neighbor


# game instance
n_state = 1000
game = DatasetSelfBreakout(
    'SelfBreakout/runs',
    'SelfBreakout/runs/0',
    binarize=0.1,
    n_state=n_state,
)
# game = DatasetAtari(
#     'BreakoutNoFrameskip-v4',
#     partial(RandomConsistentPolicy, change_prob=0.35),
#     n_state=1000,
#     save_path='results',
#     normalized_coor=True,
# )

# saliency
dmiloss = SaliencyLoss(
    game,
    c_fn_2=partial(util.hinged_mean_square_deviation, 
                   alpha_d=0.3),  # TODO: parameterize this
    frame_dev_coeff= 0.0,
    focus_dev_coeff= 50.0,
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
# micplosses.append(action_micploss)

# premise loss
net_params_path = 'ObjectRecognition/net_params/two_layer.json'
net_params = json.loads(open(net_params_path).read())
paddle_model = ModelFocusCNN(
    image_shape=(84, 84),
    net_params=net_params,
)
paddle_model.set_parameters(np.load('results/cmaes_soln/focus_self/paddle_bin.npy'))
net_params_path = 'ObjectRecognition/net_params/two_layer.json'
net_params = json.loads(open(net_params_path).read())
ball_model = ModelFocusCNN(
    image_shape=(84, 84),
    net_params=net_params,
)
ball_model.set_parameters(np.load('results/cmaes_soln/focus_self/ball_bin.npy'))
net_params_path = 'ObjectRecognition/net_params/two_layer.json'
net_params = json.loads(open(net_params_path).read())
comp_model = ModelFocusCNN(
    image_shape=(84, 84),
    net_params=net_params,
)
comp_model.set_parameters(np.load('results/cmaes_soln/focus_self/42068_40.npy'))
premise_micploss = PremiseMICPLoss(
    game,
    paddle_model,
    mi_match_coeff= 0.0,
    mi_diffs_coeff= 0.0,
    mi_valid_coeff= 0.0,
    mi_cndcp_coeff= 1.0,
    prox_dist= 0.1,
    verbose=True,
)
micplosses.append(premise_micploss)

# combine every loss together
micploss = CollectionMICPLoss(*micplosses)
loss = CombinedLoss(dmiloss, micploss)
print(loss)
loss_fn = loss.forward

# get focus lists for comparison
LIMIT = n_state
L = game.idx_offset
R = L + LIMIT
try:
    ideal_paddle = game.paddle_data.astype(float)[L:R, ...] / 84.0
except:
    ideal_paddle = np.random.rand(LIMIT, 2)
try:
    ideal_ball = game.ball_data.astype(float)[L:R, ...] / 84.0
except:
    ideal_ball = np.random.rand(LIMIT, 2)
random_focus = np.random.rand(LIMIT, 2)
paddle_model_focus = paddle_model.forward(torch.from_numpy(game.get_frame(0, LIMIT)).float())
ball_model_focus = ball_model.forward(torch.from_numpy(game.get_frame(0, LIMIT)).float())
comp_model_focus = comp_model.forward(torch.from_numpy(game.get_frame(0, LIMIT)).float())
fix_jump_focus = np.zeros((LIMIT, 2))
fix_jump_focus[[LIMIT//6, LIMIT//4, LIMIT//2]] = paddle_model_focus[[LIMIT//6, LIMIT//4, LIMIT//2]]

# plot tracking paddle
if False:
    L, R = 20, 30
    plot_focus(game, range(L, R), ideal_paddle[L:R])
    plot_focus(game, range(L, R), ideal_ball[L:R])
    plot_focus(game, range(L, R), random_focus[L:R])
    plot_focus(game, range(L, R), paddle_model_focus[L:R])
    plot_focus(game, range(L, R), ball_model_focus[L:R])
    plot_focus(game, range(L, R), comp_model_focus[L:R])

# calculate loss
print('Ideal Paddle Loss:', loss_fn(ideal_paddle[:LIMIT]))
print('Ideal Ball Loss:', loss_fn(ideal_ball[:LIMIT]))
print('Random Loss:', loss_fn(random_focus[:LIMIT]))
print('Fix Jump Loss:', loss_fn(fix_jump_focus[:LIMIT]))
print('Model Paddle Loss:', loss_fn(paddle_model_focus[:LIMIT]))
print('Model Ball Loss:', loss_fn(ball_model_focus[:LIMIT]))
print('Model Compared Loss:', loss_fn(comp_model_focus[:LIMIT]))