import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', default='a2c',
                        help='algorithm to use: a2c, ppo, evo')
    # algorithm hyperparameters
    parser.add_argument('--optim', default="RMSprop",
                        help='optimizer to use: Adam, RMSprop, Evol')
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 1e-6)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop/Adam optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999),
                        help='Adam optimizer betas (default: (0.9, 0.999))')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Adam optimizer l2 norm constant (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--return-enum', type=int, default=0,
                        help='determines what return equation to use. Default is 0, which is default return, 0 is gae, 1 is buffer, 2 is segmented')
    parser.add_argument('--return-format', type=int, default=0,
                        help='0 for default, 1 for gae, 2 for return queue')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy loss term coefficient (default: 0.01)')
    parser.add_argument('--high-entropy', type=float, default=1,
                        help='high entropy (for low frequency) term coefficient (default: 1)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--num-stack', type=int, default=4,
                        help='number of frames to stack (default: 4)')
    parser.add_argument('--num-layers', type=int, default=1,
                    help='number of layers for network')
    parser.add_argument('--factor', type=int, default=4,
                        help='decides width of the network')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='Normalized inputs for the neural network/function approximator')

    # offline learning parameters
    parser.add_argument('--grad-epoch', type=int, default=1,
                        help='number of gradient epochs in offline learning (default: 1, 4 good)')

    #PPO parameters
    parser.add_argument('--clip-param', type=float, default=0.2,
                    help='ppo clip parameter (default: 0.2)')

    # Evolution parameters
    parser.add_argument('--select-ratio', type=float, default=0.25,
                    help='percentage of population selected in evolution(default: 0.25)')
    parser.add_argument('--num-population', type=int, default=20,
                        help='size of the population (default: 20)')
    parser.add_argument('--sample-duration', type=int, default=100,
                        help='number of time steps to evaluate a subject of the population (default: 100)')
    parser.add_argument('--elitism', action='store_true', default=False,
                        help='keep the best performing networks')
    parser.add_argument('--evo-gradient', type=float, default=-1,
                        help='take a step towards the weighted mean, with weight as given, (default -1, not used)')
    # Evolution Gradient parameters
    parser.add_argument('--sample-steps', type=int, default=2000,
                    help='number of time steps to run to evaluate full population (default: 2000)')
    parser.add_argument('--grad-sample-steps', type=int, default=2000,
                    help='number of time steps to run to run ppo gradient on best performer (default: 2000)')
    parser.add_argument('--grad-lr', type=float, default=.0007,
                        help='the learning rate for the PPO steps (default -1, not used)')
    # Behavior policy parameters
    parser.add_argument('--greedy-epsilon', type=float, default=0.1,
                    help='percentage of random actions in epsilon greedy')


    # pretraining arguments
    parser.add_argument('--pretrain-iterations', type=int, default=-1,
                    help='number of time steps to run the pretrainer using PPO on optimal demonstration data, -1 means not used (default: -1)')

    # Learning settings
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=1,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=5,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--num-grad-states', type=int, default=-1,
                        help='number of forward steps used to compute gradient, -1 for not used (default: -1)')
    parser.add_argument('--buffer-steps', type=int, default=-1,
                        help='number of buffered steps in the record buffer, -1 implies it is not used (default: -1)')
    parser.add_argument('--changepoint-queue-len', type=int, default=-1,
                        help='number of steps in the queue for computing the changepoints, -1 for not used (default: -1)')
    parser.add_argument('--num-iters', type=int, default=int(2e3),
                        help='number of iterations for training (default: 2e3)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # logging settings
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 10)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--record-rollouts', default="",
                        help='path to where rollouts are recorded for this run')
    parser.add_argument('--unique-id', default="0",
                        help='a differentiator for the save path for this network')
    parser.add_argument('--save-past', type=int, default=-1,
                    help='save past, saves a new net at the interval, -1 disables, must be a multiple of save-interval (default: -1)')
    # changepoint parameters
    parser.add_argument('--changepoint-dir', default=['./runs/'], nargs='+',
                        help='directory to save changepoint related values, multiple paths for ')
    parser.add_argument('--option-dir', default=[], nargs='+',
                        help='directory to save options related to changepoint values, multiple paths for ')
    parser.add_argument('--edges', default=[''], nargs='+',
                        help='All edges currently in the graph, including the train-edge, of the form Object->Object')
    parser.add_argument('--train-edge', default='',
                        help='the edge to be trained, of the form Object->Object, for changepoints, just Object')
    parser.add_argument('--changepoint-name', default='changepoint',
                        help='name to save changepoint related values')
    parser.add_argument('--champ-parameters', default=["Paddle"], nargs='+',
                    help='parameters for champ in the order len_mean, len_sigma, min_seg_len, max_particles, model_sigma, dynamics model enum (0 is position, 1 is velocity, 2 is displacement). Pre built Paddle and Ball can be input as "paddle", "ball"')
    # TODO: add all of the CHAMP parameters, and the DP-GMM parameters
    # environmental variables
    parser.add_argument('--num-frames', type=int, default=10e4,
                        help='number of frames to use for the training set (default: 10e6)')
    parser.add_argument('--env-name', default='BreakoutNoFrameskip-v4',
                        help='environment to train on (default: BreakoutNoFrameskip-v4)')
    parser.add_argument('--train', action ='store_true', default=False,
                        help='trains the algorithm if set to true')
    # load variables
    parser.add_argument('--load-weights', default="",
                        help='path to trained model, or to data if training by imitation')
    parser.add_argument('--load-networks', default=[], nargs='+',
                        help='path to networks folder')

    args = parser.parse_args()

    if args.champ_parameters[0] == "Paddle":
        args.champ_parameters = [3, 5, 1, 100, 100, 2, 1e-1, 0]
    if args.champ_parameters[0] == "Ball": 
        args.champ_parameters = [15, 10, 2, 100, 100, 2, 1, 0] 
    else:
        args.champ_parameters = [int(p) for p in args.champ_parameters]

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args

def get_edge(edge):
    head = edge.split("->")[0]
    tail = edge.split("->")[1]
    head = head.split(",")
    return head, tail[0]
