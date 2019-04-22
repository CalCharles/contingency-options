from ObjectRecognition.model import (
    ModelFocusCNN, ModelCollectionDAG,
    load_param)
import os, json
from Environments.state_definition import GetRaw, load_states
from arguments import get_args



if __name__ == '__main__':
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
    model.add_model('paddle', paddle_model, [])
    model.add_model('ball', ball_model, ['paddle'])
    print(model)
    state_function = GetRaw()
    states, resps, raws, dumps = load_states(state_function.get_state, pth, use_raw = True)
    dumps = model.forward(raws, ret_numpy=True)
    focus_dumps = [{} for _ in range(len(states))]
    for key in dumps:
        for i, val in enumerate(dumps[key]):
            focus_dumps[i][key] = val
    focus_dumps_file = open(os.path.join(args.record_rollouts, "focus_dumps.txt", 'w'))
    for factor_state in focus_dumps:
        for key in factor_state.keys():
            focus_dumps_file.write(key + ":" + " ".join(factor_state[key]) + "\t") # TODO: attributes are limited to single floats
        focus_dumps_file.write("\n") # TODO: recycling does not stop object dumping
    focus_dumps_file.close()