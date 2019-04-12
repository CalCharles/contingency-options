import os, pickle
import numpy as np
from arguments import get_args
from file_management import load_from_pickle, save_to_pickle, get_edge, get_cp_models_from_dict, read_obj_dumps, get_individual_data
from RewardFunctions.dataTransforms import arg_transform
from RewardFunctions.changepointClusterModels import MultipleCluster, BayesianGaussianMixture
from RewardFunctions.changepointDeterminers import determiners
from RewardFunctions.changepointCorrelation import ChangepointModels
from RewardFunctions.changepointReward import reward_forms
import ChangepointDetection.DynamicsModels as DynamicsModels
import ChangepointDetection.CHAMP as CHAMP
from ChangepointDetection.CHAMP import CHAMPDetector, CHAMP_parameters

# Example usages:
# Learn paddle with hacked reward
#  python get_reward.py --record-rollouts data/random/ --changepoint-dir data/optgraph/ --train-edge "Action->Paddle" --transforms SVel SCorAvg --determiner overlap --reward-form markov --segment --train --num-stack 2
def load_from_pickle(pth):
    with open(pth, 'rb') as fid:
        save_dict = pickle.load(fid)
    return save_dict

if __name__ == "__main__":
    args = get_args()
    # Required arguments:
        # record_rollouts
        # changepoint-dir
        # train-edge
        # transforms
        # dp-gmm (if not included, default)
        # determiner (TODO: add determiner args later)
        # window (if used)
        # reward-form
        # train (used for trainable rewards)
        # segment

    dataset_path = args.record_rollouts
    changepoints_path = args.record_rollouts # these are the same for creating rewards
    head, tail = get_edge(args.train_edge)
    cp_dict = load_from_pickle(os.path.join(changepoints_path, "changepoints-" + head + ".pkl"))
    changepoints, models = get_cp_models_from_dict(cp_dict)
    obj_dumps = read_obj_dumps(dataset_path)

    trajectory = get_individual_data(head, obj_dumps, pos_val_hash=1)
    # TODO: automatically determine if correlate pos_val_hash is 1 or 2
    # TODO: multiple tail support
    if tail[0] == "Action":
        correlate_trajectory = get_individual_data(tail[0], obj_dumps, pos_val_hash=2)
    else:
        correlate_trajectory = get_individual_data(tail[0], obj_dumps, pos_val_hash=1)

    combined = np.concatenate([trajectory, correlate_trajectory], axis=1)

    changepoint_model = load_from_pickle(os.path.join(changepoints_path, "detector-" + head + ".pkl"))
    # paddle uses "SVel", "SCorAvg"
    # first ball uses: "WProx"
    # second ball uses: "WProx", "WVel", 
    transforms = [arg_transform[tform]() for tform in args.transforms]

    # TODO: other cluster models?
    clusters = MultipleCluster(args, BayesianGaussianMixture)

    # paddle uses "overlap", ball uses "prox", "proxVel"
    determiner = determiners[args.determiner]()# PureOverlapDeterminer(overlap_ratio = .95, min_cluster= 7)
    option_determiner_model = ChangepointModels(args, changepoint_model, transforms, clusters, determiner)
    option_determiner_model.changepoint_statistics(models, changepoints, trajectory, correlate_trajectory)
    
    try:
        os.makedirs(os.path.join(args.changepoint_dir, args.train_edge))
    except OSError:
        pass # folder already created
    print(args.changepoint_dir)
    for i in range(option_determiner_model.determiner.num_mappings):
        reward_function = reward_forms[args.reward_form](option_determiner_model, args, i)
        if args.train:
            reward_function.generate_training_set(combined, models, np.array(changepoints))
            reward_function.train_rewards(20000)

        save_to_pickle(os.path.join(args.changepoint_dir, args.train_edge, "reward__function__" + str(i) +"__rwd.pkl"), reward_function)
