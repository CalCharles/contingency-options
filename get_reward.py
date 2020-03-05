import os, pickle
import numpy as np
import torch
from arguments import get_args
from file_management import load_from_pickle, save_to_pickle, get_edge, get_cp_models_from_dict, read_obj_dumps, get_individual_data
from RewardFunctions.dataTransforms import arg_transform
from RewardFunctions.changepointClusterModels import MultipleCluster, BayesianGaussianMixture, cluster_models
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
    torch.cuda.set_device(args.gpu)
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
    # atari action->paddle: python get_reward.py --record-rollouts data/atarirandom/ --changepoint-dir data/atarigraph/ --train-edge "Action->Paddle" --transforms SVel SCorAvg --determiner overlap --reward-form markov --segment --train --num-stack 2 --focus-dumps-name focus_dumps.txt --dp-gmm atari
    # python get_reward.py --record-rollouts data/atarirandom/ --changepoint-dir data/atarigraph/ --train-edge "Action->Paddle" --transforms WProx --determiner prox --reward-form changepoint --num-stack 1 --focus-dumps-name focus_dumps.txt --dp-gmm atari
    # python get_reward.py --record-rollouts data/ataripaddle/ --changepoint-dir data/atarigraph/ --train-edge "Paddle->Ball" --transforms WProx --determiner prox --reward-form changepoint --num-stack 1 --focus-dumps-name focus_dumps.txt --dp-gmm atariball --period 5
    # python get_reward.py --record-rollouts data/pusherrandom/ --changepoint-dir data/fullpusher/ --train-edge "Action->Gripper" --transforms SVel SCorAvg --determiner overlap --reward-form markov --segment --train --num-stack 2 --gpu 1
    # python get_reward.py --record-rollouts data/extragripper/ --changepoint-dir data/pushergraphvec/ --train-edge "Gripper->Block" --transforms SProxVel --determiner merged --reward-form changepoint --segment --num-stack 2 --gpu 1 --cluster-model FDPGMM --period 9 --dp-gmm block --min-cluster 5
    # python get_reward.py --record-rollouts data/pusherrandom/ --changepoint-dir data/fullpusher/ --train-edge "Action->Gripper" --transforms SVel SCorAvg --determiner overlap --reward-form markov --segment --train --num-stack 2 --gpu 1 > pusher/reward_training.txt
    dataset_path = args.record_rollouts
    changepoints_path = args.record_rollouts # these are the same for creating rewards
    head, tail = get_edge(args.train_edge)
    cp_dict = load_from_pickle(os.path.join(changepoints_path, "changepoints-" + head + ".pkl"))
    changepoints, models = get_cp_models_from_dict(cp_dict)
    obj_dumps = read_obj_dumps(dataset_path, i=-1, rng=args.num_iters, filename = args.focus_dumps_name)

    trajectory = get_individual_data(head, obj_dumps, pos_val_hash=1)
    # TODO: automatically determine if correlate pos_val_hash is 1 or 2
    # TODO: multiple tail support
    if tail[0] == "Action":
        correlate_trajectory = get_individual_data(tail[0], obj_dumps, pos_val_hash=2)
        new_ct = np.zeros((len(correlate_trajectory), int(np.max(correlate_trajectory))+1))
        hot_idxes = np.array(list(range(len(correlate_trajectory)))), correlate_trajectory.astype(int)
        new_ct[np.array(list(range(len(correlate_trajectory)))), np.squeeze(correlate_trajectory.astype(int))] = 1
        correlate_trajectory = new_ct
    else:
        correlate_trajectory = get_individual_data(tail[0], obj_dumps, pos_val_hash=1)

    combined = np.concatenate([trajectory, correlate_trajectory], axis=1)
    # print([model.data for model in reversed(models)])


    changepoint_model = load_from_pickle(os.path.join(changepoints_path, "detector-" + head + ".pkl"))
    # paddle uses "SVel", "SCorAvg"
    # first ball uses: "WProx"
    # second ball uses: "WProx", "WVel", 
    transforms = [arg_transform[tform]() for tform in args.transforms]

    # TODO: other cluster models?
    clusters = MultipleCluster(args, cluster_models[args.cluster_model])

    # paddle uses "overlap", ball uses "prox", "proxVel"
    determiner =     determiners[args.determiner](prox_distance = args.period, overlap_ratio=.5, min_cluster=args.min_cluster) # reusing period to define minimum distance# PureOverlapDeterminer(overlap_ratio = .95, min_cluster= 7)
    option_determiner_model = ChangepointModels(args, changepoint_model, transforms, clusters, determiner)
    option_determiner_model.changepoint_statistics(models, changepoints, trajectory, correlate_trajectory)

    try:
        os.makedirs(os.path.join(args.changepoint_dir, args.train_edge))
    except OSError:
        pass # folder already created
    print(args.changepoint_dir)
    reward_fns = []
    for i in range(option_determiner_model.determiner.num_mappings):
        reward_function = reward_forms[args.reward_form](option_determiner_model, args, i)
        if args.train:
            reward_function.generate_training_set(combined, models, np.array(changepoints))
            reward_function.train_rewards(20000)
        save_to_pickle(os.path.join(args.changepoint_dir, args.train_edge, "reward__function__" + str(i) +"__rwd.pkl"), reward_function)
        reward_fns.append(reward_function)
    # if args.train:
    #     minvar = np.min(np.max([rf.markovModel.variance.tolist() for rf in reward_fns]), axis=0)
    #     print(minvar)
    #     for i, rf in enumerate(reward_fns):
    #         rf.setvar(minvar)
    #         save_to_pickle(os.path.join(args.changepoint_dir, args.train_edge, "reward__function__" + str(i) +"__rwd.pkl"), rf)