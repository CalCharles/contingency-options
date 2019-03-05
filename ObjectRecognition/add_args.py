from __future__ import division, absolute_import, print_function


def add_changepoint_argument(parser):
    parser.add_argument('--champ', choices=['ball', 'paddle'],
                        help='parameters for CHAMP')


def add_dataset_argument(parser):
    parser.add_argument('--n_state', type=int, default=1000,
                        help='number of states in an episode')
    parser.add_argument('--binarize', type=float, default=None,
                        help='game binarize threshold')
    parser.add_argument('--offset_fix', type=int, default=None,
                        help='episode number (if dataset allow)')


def add_optimizer_argument(parser):
    parser.add_argument('--niter', type=int, default=40,
                        help='number of training iterations')
    parser.add_argument('--popsize', type=int, default=20,
                        help='CMA-ES population size')
    parser.add_argument('--nproc', type=int, default=1,
                        help='number of processors')
    parser.add_argument('--cheating', choices=['ball', 'paddle', 'gaussian'],
                        help='plot model filter')


def add_model_argument(parser):
    parser.add_argument('--boost', type=str, nargs=2, default=None,
                        metavar=('NET-PARAMS', 'WEIGHT'),
                        help='train boost on top')
    parser.add_argument('--prior', action='store_true', default=False,
                        help='use focus prior filter')  # move into net_params?
    parser.add_argument('--argmax_mode',
                        choices=['first', 'rand'], default='first',
                        help='argmax mode to choose focus coordinate')


def add_loss_argument(parser):
    parser.add_argument('--saliency', type=float, nargs=4, default=None,
                        metavar=('FRAME-DEV', 'FOCUS-DEV', 'FRAME-VAR', 
                                 'BELIEF-DEV'),
                        help='coefficients for saliency loss (3)')
    parser.add_argument('--hinge_dist', type=float, default=0.2,
                        help='hinge distance for focus deviation')
    parser.add_argument('--action_micp', type=float, nargs=2, default=None,
                        metavar=('MATCH', 'DIFFS'),
                        help='coefficients for action MICP loss (2)')
    parser.add_argument('--premise_micp', type=float, nargs=4, default=None,
                        metavar=('MATCH', 'DIFFS', 'VALID', 'PROX-DIST'),
                        help='coefficients for premise MICP loss (4)')
    parser.add_argument('--premise_path', type=str,
                        default='results/cmaes_soln/focus_self/paddle.npy',
                        help='path to network weight for premise recognition')
    parser.add_argument('--premise_net', type=str,
                        default='ObjectRecognition/net_params/two_layer.json',
                        help='path to network params for premise recognition')