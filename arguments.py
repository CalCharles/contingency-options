import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    # parser.add_argument('--algo', default='a2c',
    #                     help='algorithm to use: a2c, ppo, evo')
    # # optimization hyperparameters
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 1e-6)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop/Adam optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999),
                        help='Adam optimizer betas (default: (0.9, 0.999))')
    parser.add_argument('--weight-decay', type=float, default=0.00,
                        help='Adam optimizer l2 norm constant (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    # cost function hyperparameters
    parser.add_argument('--return-enum', type=int, default=0,
                        help='determines what return equation to use. Default is 0, which is default return, 0 is gae, 1 is buffer, 2 is segmented')
    parser.add_argument('--return-format', type=int, default=0,
                        help='0 for default, 1 for gae, 2 for return queue')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=1e-7,
                        help='entropy loss term coefficient (default: 1e-7)')
    parser.add_argument('--high-entropy', type=float, default=1,
                        help='high entropy (for low frequency) term coefficient (default: 1)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    # model hyperparameters
    parser.add_argument('--num-layers', type=int, default=1,
                    help='number of layers for network. When using basis functions, defines independence relations (see ReinforcementLearning.basis_models.py)')
    parser.add_argument('--factor', type=int, default=4,
                        help='decides width of the network')
    parser.add_argument('--optim', default="Adam",
                        help='optimizer to use: Adam, RMSprop, Evol')
    parser.add_argument('--init-form', default="uni",
                    help='initialization to use: uni, xnorm, xuni, eye')
    parser.add_argument('--model-form', default="",
                        help='choose the model form, which is defined in ReinforcementLearning.models')
    # state hyperparameters
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='Normalized inputs for the neural network/function approximator')
    parser.add_argument('--num-stack', type=int, default=4,
                        help='number of frames to stack (default: 4)')

    # distributional RL parameters
    parser.add_argument('--value-bounds', type=float, nargs=2, default=(0, 10),
                        help='bounds for the possible value of a state (default: (0, 10))')
    parser.add_argument('--num-value-atoms', type=int, default=51,
                        help='number of atoms in distributional RL (default: 51)')
    # distributional regularization parameters
    parser.add_argument('--dist-interval', type=int, default=-1,
                        help='decides how often distributional interval is computed')
    parser.add_argument('--exp-beta', type=float, default=0.1,
                        help='beta value in exponential distribution')
    parser.add_argument('--dist-coef', type=float, default=1e-5,
                        help='the coefficient used for determining the loss value of the distribution')


    # offline learning parameters
    parser.add_argument('--grad-epoch', type=int, default=1,
                        help='number of gradient epochs in offline learning (default: 1, 4 good)')

    #PPO parameters
    parser.add_argument('--clip-param', type=float, default=0.2,
                    help='ppo clip parameter (default: 0.2)')

    # Evolution parameters TODO: evolution not implemented at the moment
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
    # basis function parameters
    parser.add_argument('--period', type=float, default=1,
                help='length of period over which fourier basis is applied')
    parser.add_argument('--scale', type=float, default=1,
                help='scaling term for magnitudes, which can be useful in multilayer to exacerbate differences')

    # Behavior policy parameters
    parser.add_argument('--greedy-epsilon', type=float, default=0.1,
                    help='percentage of random actions in epsilon greedy')
    parser.add_argument('--min-greedy-epsilon', type=float, default=0.1,
                    help='minimum percentage of random actions in epsilon greedy (if decaying)')
    parser.add_argument('--greedy-epsilon-decay', type=float, default=-1,
                    help='greedy epsilon decays by half every n updates (-1 is for no use)')
    parser.add_argument('--behavior-policy', default='',
                        help='defines the behavior policy, as defined in BehaviorPolicies.behavior_policies')


    # pretraining arguments TODO: not implemented
    parser.add_argument('--pretrain-iterations', type=int, default=-1,
                    help='number of time steps to run the pretrainer using PPO on optimal demonstration data, -1 means not used (default: -1)')
    # Reinforcement model settings
    parser.add_argument('--optimizer-form', default="",
                        help='choose the optimizer form, which is defined in ReinforcementLearning.learning_algorithms')
    parser.add_argument('--state-forms', default=[""], nargs='+',
                    help='the different relational functions, which are defined in Environment.state_definition')
    parser.add_argument('--state-names', default=[""], nargs='+',
                    help='should match the number of elements in state-forms, contains the names of nodes used')
    # Learning settings
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=1,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=5,
                        help='number of forward steps before update (default: 5)')
    parser.add_argument('--num-update-model', type=int, default=3,
                        help='number of gradient steps before switching options (default: 3)')
    parser.add_argument('--num-grad-states', type=int, default=-1,
                        help='number of forward steps used to compute gradient, -1 for not used (default: -1)')
    parser.add_argument('--buffer-steps', type=int, default=-1,
                        help='number of buffered steps in the record buffer, -1 implies it is not used (default: -1)')
    parser.add_argument('--changepoint-queue-len', type=int, default=30,
                        help='number of steps in the queue for computing the changepoints')
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
                        help='directory to save data when adding edges')
    parser.add_argument('--save-recycle', type=int, default=-1,
                        help='only saves the last n timesteps (-1 if not used)')
    parser.add_argument('--record-rollouts', default="",
                        help='path to where rollouts are recorded (when adding edges, where data was recorded to compute min/max)')
    parser.add_argument('--changepoint-dir', default='./data/optgraph/',
                        help='directory to save/load option chain')
    parser.add_argument('--unique-id', default="0",
                        help='a differentiator for the save path for this network')
    parser.add_argument('--save-past', type=int, default=-1,
                    help='save past, saves a new net at the interval, -1 disables, must be a multiple of save-interval (default: -1)')
    parser.add_argument('--save-models', action ='store_true', default=False,
                        help='Saves environment and models to option chain directory if true')
    # Option Chain Parameters
    parser.add_argument('--base-node', default="Action",
                        help='The name of the lowest node in the option chain (generally should be Action)')

    # changepoint parameters
    parser.add_argument('--past-data-dir', default='',
                        help='directory to load data for computing minmax')
    parser.add_argument('--segment', action='store_true', default=False,
                    help='if true, the reward function gives reward for a full segment, while if false, will apply a single sparse reward')
    parser.add_argument('--transforms', default=[''], nargs='+',
                    help='Different transforms to be used to reduce a segment or window to a single value. Options are in RewardFunctions.dataTransforms.py')
    parser.add_argument('--train-edge', default='',
                        help='the edge to be trained, of the form Object->Object, for changepoints, just Object')
    parser.add_argument('--determiner', default='',
                        help='defines the determiner to use, using strings as defined in RewardFunctions.changepointDeterminers')
    parser.add_argument('--reward-form', default='',
                        help='defines the kind of reward function to use, as defined in RewardFunctions.changepointReward, also: dense, x, bounce')
    parser.add_argument('--changepoint-name', default='changepoint',
                        help='name to save changepoint related values')
    parser.add_argument('--champ-parameters', default=["Paddle"], nargs='+',
                    help='parameters for champ in the order len_mean, len_sigma, min_seg_len, max_particles, model_sigma, dynamics model enum (0 is position, 1 is velocity, 2 is displacement). Pre built Paddle and Ball can be input as "paddle", "ball"')
    parser.add_argument('--window', type=int, default=3,
                help='A window over which to compute changepoint statistics')
    # environmental variables
    parser.add_argument('--num-frames', type=int, default=10e4,
                        help='number of frames to use for the training set (default: 10e6)')
    parser.add_argument('--env-name', default='BreakoutNoFrameskip-v4',
                        help='environment to train on (default: BreakoutNoFrameskip-v4)')
    parser.add_argument('--train', action ='store_true', default=False,
                        help='trains the algorithm if set to true')
    # load variables
    parser.add_argument('--load-weights', action ='store_true', default=False,
                        help='load the options for the existing network')
    # parser.add_argument('--load-networks', default=[], nargs='+',
    #                     help='load weights from the network')
    # DP-GMM parameters
    parser.add_argument('--dp-gmm', default=["default"], nargs='+',
                    help='parameters for dirichlet process gaussian mixture model, in order number of components, maximum iteration number, prior, covariance type and covariance prior')
            


    args = parser.parse_args()
    if args.dp_gmm[0] == 'default':
        args.dp_gmm = [10, 6000, 100, 'diag', 1e-10]
    if args.champ_parameters[0] == "Paddle":
        args.champ_parameters = [3, 5, 1, 100, 100, 2, 1e-1, 0]
    elif args.champ_parameters[0] == "Ball": 
        args.champ_parameters = [15, 10, 2, 100, 100, 2, 1, 0] 
    else:
        args.champ_parameters = [float(p) for p in args.champ_parameters]

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args

def get_edge(edge):
    head = edge.split("->")[0]
    tail = edge.split("->")[1]
    head = head.split(",")
    return head, tail[0]
