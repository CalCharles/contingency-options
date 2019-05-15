from AtariEnvironments.focus_atari import FocusAtariEnvironment
from SelfBreakout.breakout_screen import RandomPolicy, RotatePolicy
from ObjectRecognition.model import (
    ModelFocusCNN, ModelCollectionDAG,
    load_param, util)
import json, sys, cv2, torch
from Models.models import pytorch_model

if __name__ == '__main__':
    torch.cuda.set_device(1)
    paddle_model_net_params_path = 'ObjectRecognition/net_params/attn_softmax.json'
    net_params = json.loads(open(paddle_model_net_params_path).read())
    params = load_param('ObjectRecognition/models/atari/paddle_bin_smooth.pth')
    paddle_model = ModelFocusCNN(
        image_shape=(84, 84),
        net_params=net_params,
        binarize = 0.01
    )
    paddle_model.set_parameters(params)
    ball_model_net_params_path = 'ObjectRecognition/net_params/attn_softmax.json'
    net_params = json.loads(open(ball_model_net_params_path).read())
    params = load_param('ObjectRecognition/models/atari/42531_2_smooth.pth')
    ball_model = ModelFocusCNN(
        image_shape=(84, 84),
        net_params=net_params,
        binarize = 0.01
    )
    ball_model.set_parameters(params)
    model = ModelCollectionDAG()
    model.add_model('Paddle', paddle_model, [], augment_fn=util.RemoveMeanMemory(nb_size=(3, 9)))
    model.add_model('Ball', ball_model, ['Paddle'])

    screen = FocusAtariEnvironment(model, "BreakoutNoFrameskip-v0", 1, 0, sys.argv[1])
    screen.set_save(0, sys.argv[1], -1)
    policy = RotatePolicy(4, 3)
    # policy = BouncePolicy(4)
    for i in range(2000):
        action = policy.act(screen)
        raw_state, factor_state, done = screen.step(pytorch_model.wrap(action))
        raw_state[int(factor_state['Paddle'][0][0]), :] = 255
        raw_state[:, int(factor_state['Paddle'][0][1])] = 255
        cv2.imshow('frame',raw_state)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("done")

