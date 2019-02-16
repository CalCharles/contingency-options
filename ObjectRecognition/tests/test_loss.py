from functools import partial
import matplotlib.pyplot as plt
from SelfBreakout.breakout_screen import RandomConsistentPolicy
from ObjectRecognition.dataset import DatasetSelfBreakout, DatasetAtari
# from ObjectRecognition.report import plot_focus
from ObjectRecognition.model import ModelFocusCNN
from ObjectRecognition.loss import *
from ObjectRecognition.util import extract_neighbor
from ObjectRecognition.main_report import plot_focus


# game instance
game = DatasetSelfBreakout(
    'SelfBreakout/runs',
    'SelfBreakout/runs/0',
)
# game = DatasetAtari(
#     'BreakoutNoFrameskip-v4',
#     partial(RandomConsistentPolicy, change_prob=0.35),
#     n_state=1000,
#     save_path='results',
#     normalized_coor=True,
# )

# model instance(s)
net_params = {
    'filter': 1,
    'channel': [1],
    'kernel_size': [8],
    'stride': [1],
    'padding': [7],
    'activation_fn': ['ReLU'],
}
model = ModelFocusCNN(
    image_shape=(84, 84),
    net_params=net_params,
)
model.set_parameters(np.load('results/cmaes_soln/focus/paddle.npy'))

# loss instances
dmiloss = SaliencyLoss(
    game,
    frame_dev_coeff= 10.0,
    focus_dev_coeff= 0.2,
    frame_var_coeff= 0.02,
    verbose=True,
)
action_micploss = ActionMICPLoss(
    game,
    mi_match_coeff= 1.2,
    mi_diffs_coeff= 0.1,
    verbose=True,
)
premise_micploss = PremiseMICPLoss(
    game,
    model,
    mi_match_coeff= 1.0,
    mi_diffs_coeff= 0.2,
    mi_valid_coeff= 0.1,
    prox_dist= 0.1,
    verbose=True,
)
micploss = CollectionMICPLoss(action_micploss, premise_micploss)
loss = CombinedLoss(dmiloss, micploss)
print(loss)
loss_fn = loss.forward

# random focus points to test against
LIMIT = 1000
try:
    ideal_paddle = game.paddle_data.astype(float)[:LIMIT, ...] / 84.0
except:
    ideal_paddle = np.random.rand(LIMIT, 2)
random_focus = np.random.rand(LIMIT, 2)

# plot tracking paddle
if False:
    L, R = 20, 30
    plot_focus(game, range(L, R), ideal_paddle[L:R])
    plot_focus(game, range(L, R), random_focus[L:R])

# calculate DMI
print('Ideal Paddle DMI Loss:', loss_fn(ideal_paddle[:LIMIT]))
print('Random Paddle DMI Loss:', loss_fn(random_focus[:LIMIT]))
