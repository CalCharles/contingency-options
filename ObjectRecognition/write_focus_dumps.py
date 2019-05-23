from collections import deque
from file_management import load_from_pickle, get_edge, read_obj_dumps
import glob, os, cv2
import argparse
from Models.models import pytorch_model
from ObjectRecognition.model import (
    ModelFocusCNN, ModelCollectionDAG,
    load_param, util)
from SelfBreakout.focus_screen import FocusEnvironment
import json
import torch
from AtariEnvironments.focus_atari import FocusAtariEnvironment
from RewardFunctions.dummy_rewards import RawReward
import numpy as np
import imageio as imio


if __name__ == '__main__':
    # python write_focus_dumps.py fullrandom results/cmaes_soln/focus_self --params-name paddle
    parser = argparse.ArgumentParser(description='Train object recognition')
    parser.add_argument('dataset', type=str,
                        help='base directory to save results')
    parser.add_argument('model_dir', type=str,
                        help='base directory with models')
    parser.add_argument('--params-name', type=str, default="attn_softmax",
                        help='base directory to save results')
    parser.add_argument('--length', type=int, default=50000,
                        help='number of records to load')
    parser.add_argument('--gpu', type=int, default=0,
                        help='the gpu to run on')
    parser.add_argument('--ball', action='store_true', default=False,
                        help='use if we need the ball model')  # move into net_params?
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use if we have cuda')  # move into net_params?
    args = parser.parse_args()
    # paddle_model_net_params_path = 'ObjectRecognition/net_params/attn_base.json'
    paddle_model_net_params_path = 'ObjectRecognition/net_params/%s.json'%args.params_name
    net_params = json.loads(open(paddle_model_net_params_path).read())
    # params = load_param('ObjectRecognition/models/self/paddle_bin_long_smooth.pth')
    params = load_param(os.path.join(args.model_dir,'%s.pth'%"paddle"))
    paddle_model = ModelFocusCNN(
        image_shape=(84, 84),
        net_params=net_params,
        binarize=0.0
    )
    paddle_model.set_parameters(params)
    # ball_model_net_params_path = 'ObjectRecognition/net_params/attn_base.json'
    if args.ball:
        ball_model_net_params_path = 'ObjectRecognition/net_params/%s.json'%args.params_name
        net_params = json.loads(open(ball_model_net_params_path).read())
        # params = load_param('ObjectRecognition/models/self/ball_bin_long_smooth.pth')
        params = load_param(os.path.join(args.model_dir,'%s.pth'%"ball"))
        ball_model = ModelFocusCNN(
            image_shape=(84, 84),
            net_params=net_params,
            binarize=0.0
        )
        ball_model.set_parameters(params)
    model = ModelCollectionDAG()
    model.add_model('Paddle', paddle_model, [], augment_fn=util.RemoveMeanMemory(nb_size=(8, 8)))
    if args.ball:
        f1 = util.LowIntensityFiltering(8.0)
        f2 = util.JumpFiltering(3, 0.05)
        def f(x, y):
            return f2(x, f1(x, y))
            # model.add_model('train', r_model, ['premise'], augment_pt=f)

        model.add_model('Ball', ball_model, ['Paddle'])#, augment_pt=f)#,augment_pt=util.JumpFiltering(2, 0.05))
    if args.cuda:
        torch.cuda.set_device(1)
        model = model.cuda()
    raw_files = deque(maxlen=args.length)
    for root, dirs, files in os.walk(args.dataset, topdown=False):
        dirs.sort(key=lambda x: int(x))
        print(args.dataset, dirs)
        for d in dirs:
            try:
                for p in [os.path.join(args.dataset, d, "state" + str(i) + ".png") for i in range(2000)]:
                    raw_files.append(imio.imread(p))
            except OSError as e:
                # reached the end of the file
                pass
    dumps = read_obj_dumps(args.dataset, i=-1, rng = args.length, filename="object_dumps.txt")
    for i, (dump, raw_state) in enumerate(zip(dumps, raw_files)):
        factor_state = model.forward(pytorch_model.wrap(raw_state, cuda=False).unsqueeze(0).unsqueeze(0), ret_numpy=True)
        for key in factor_state.keys():
            factor_state[key] *= 84
            factor_state[key] = (np.squeeze(factor_state[key]), (1.0,))
        factor_state['Action'] = dump['Action']
        if i != 0:
            object_dumps = open(os.path.join(args.dataset, "focus_dumps.txt"), 'a')
        else:
            object_dumps = open(os.path.join(args.dataset, "focus_dumps.txt"), 'w') # create file if it does not exist
        for key in factor_state.keys():
            writeable = list(factor_state[key][0]) + list(factor_state[key][1])
            object_dumps.write(key + ":" + " ".join([str(fs) for fs in writeable]) + "\t") # TODO: attributes are limited to single floats
        object_dumps.write("\n") # TODO: recycling does not stop object dumping

