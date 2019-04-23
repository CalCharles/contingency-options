from ObjectRecognition.model import (
    ModelFocusCNN, ModelCollectionDAG,
    load_param, util)
import os, json
from Models.models import pytorch_model 
import numpy as np
from Environments.state_definition import GetRaw, load_states, GetState
from arguments import get_args



if __name__ == '__main__':
    # python generate_focus_dumps.py --record-rollouts data/integrationpaddle/
    args = get_args()
    paddle_model_net_params_path = 'ObjectRecognition/net_params/two_layer.json'
    net_params = json.loads(open(paddle_model_net_params_path).read())
    params = load_param('ObjectRecognition/models/paddle.npy')
    paddle_model = ModelFocusCNN(
        image_shape=(84, 84),
        net_params=net_params,
    )
    paddle_model.set_parameters(params)
    ball_model_net_params_path = 'ObjectRecognition/net_params/two_layer.json'
    net_params = json.loads(open(ball_model_net_params_path).read())
    params = load_param('ObjectRecognition/models/ball.npy')
    ball_model = ModelFocusCNN(
        image_shape=(84, 84),
        net_params=net_params,
    )
    ball_model.set_parameters(params)
    model = ModelCollectionDAG()
    model.add_model('Paddle', paddle_model, [], augment_fn=util.remove_mean)
    model.add_model('Ball', ball_model, ['Paddle'])
    print(model)
    state_function = GetState('Action', state_forms=[("Action", "feature")])
    states, resps, raws, dumps = load_states(state_function.get_state, args.record_rollouts, use_raw = True)
    print(states)
    raws = pytorch_model.wrap(np.array(raws), cuda=False).unsqueeze(1)
    dumps = model.forward(raws, ret_numpy=True)
    focus_dumps = [{} for _ in range(len(states))]
    for key in dumps:
        for i, val in enumerate(dumps[key]):
            focus_dumps[i][key] = val
    focus_dumps_file = open(os.path.join(args.record_rollouts, "focus_dumps.txt"), 'w')
    for action, factor_state in zip(states, focus_dumps):
        for key in factor_state.keys():
            focus_dumps_file.write(key + ":" + " ".join([str(val * 84) for val in factor_state[key]]) + " 1.0" + "\t") # TODO: attributes are limited to single floats
        focus_dumps_file.write("Action:" + "0.0 0.0 " + str(action[0]) + "\t") # TODO: attributes are limited to single floats
        focus_dumps_file.write("\n") # TODO: recycling does not stop object dumping
    focus_dumps_file.close()