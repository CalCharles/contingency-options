# time to next changepoint
from file_management import load_from_pickle
import os
import numpy as np
from Models.models import pytorch_model

rollouts = load_from_pickle(os.path.join("data", "bouncetest", "rollouts.pkl"))
reward_indexes = rollouts.rewards.sum(dim=0).nonzero()
print(len(reward_indexes))
total = rollouts.last_step
indexes = np.array(pytorch_model.unwrap(reward_indexes.squeeze()).tolist() + [total])
print(indexes)
std = np.sqrt(np.var(indexes[1:] - indexes[:-1]))
mean = np.median(indexes[1:] - indexes[:-1])
print("diffs", mean, indexes[1:] - indexes[:-1], (indexes[1:] - indexes[:-1] - mean), (indexes[1:] - indexes[:-1] - mean)/std)
option_indexes = [rollouts.rewards[oidx].nonzero() for oidx in range(len(rollouts.rewards))]
# print(reward_indexes, option_indexes)