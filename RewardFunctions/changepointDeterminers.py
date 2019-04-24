import numpy as np
from collections import Counter
from file_management import default_value_arg


class ChangepointDeterminer():
    def __init__(self, **kwargs):
        # TODO: change init to take in args, and define based on args
        pass

    def fit_narrow_modes(self, changepoint_models, mode_models, assigned_modes):
        '''
        determines the clusters of modes that are appropriate (groups of modes that seem to produce like behavior)
        changepoint models are the models for each segment
        mode models are the models for each mode (cluster)
        assigned modes are the assignment of mode for each segment/window in the changepoint models
        '''
        pass


    def collapse_assignments(self, mode_assignments):
        '''
        transforms a mode assignment vector into a single mode assignment that represents a (believed) 
        reproducible option.
        '''
        pass

class PureOverlapDeterminer(ChangepointDeterminer):
    '''
    finds the ``overlap'' between a pair of different modes, and returns pure overlap
    '''
    def __init__(self, **kwargs):
        self.overlap_ratio = default_value_arg(kwargs, 'overlap_ratio', .75)
        self.min_cluster = default_value_arg(kwargs, 'min_cluster', 7)

    def fit_narrow_modes(self, changepoint_models, mode_models, assigned_modes):
        overlap = dict()
        totals = Counter()
        for modes in assigned_modes:
            if modes[0] in overlap:
                overlap[modes[0]][modes[1]] += 1
                totals[modes[0]] += 1
            else:
                overlap[modes[0]] = Counter()
                overlap[modes[0]][modes[1]] += 1
                totals[modes[0]] += 1

        pure_ratio = dict()

        for m in overlap.keys():
            print(m, [overlap[m][k] for k in overlap[m].keys()])
            pure_ratio[m] = max([overlap[m][k] for k in overlap[m].keys()]) / totals[m]
        print(pure_ratio, overlap)
        self.pure_clusters = set()
        self.mixed_clusters = set()
        self.empty_clusters = set()
        keys = list(overlap.keys())
        keys.sort()
        self.key_mapping = dict()
        i = 1
        for m in keys:
            if totals[m] < self.min_cluster:
                self.empty_clusters.add(m)
            elif pure_ratio[m] > self.overlap_ratio:
                self.pure_clusters.add(m)
                self.key_mapping[m] = i
                i += 1
            else:
                self.mixed_clusters.add(m)
                self.key_mapping[m] = 0
        print(self.mixed_clusters, self.pure_clusters, i)
        self.num_mappings = i

    def collapse_assignments(self, assigned_modes):
        mode_assignments = []
        # print(assigned_modes.shape)
        for am in assigned_modes:
            # print(am)
            if am[0] in self.empty_clusters:
                mode_assignments.append(-1)
            else:
                mode_assignments.append(self.key_mapping[am[0]])
        return np.array(mode_assignments)

class ProximityDeterminer(ChangepointDeterminer):
    '''
    finds modes based on whether they are proximal, assuming that the data is clustered on pairwise proximity
    '''
    def __init__(self, **kwargs):
        self.prox_distance = default_value_arg(kwargs, 'prox_distance', 6) # TODO: Set proximal distance based on ...
        self.min_cluster = default_value_arg(kwargs, 'min_cluster', 7)

    def fit_narrow_modes(self, changepoint_models, mode_models, assigned_modes):
        # find clusters with at least min_cluster elements 
        seen_modes = Counter()
        for mode in assigned_modes:
            seen_modes[mode[0]] += 1
        self.used_clusters = set()
        for k in seen_modes.keys():
            if seen_modes[k] > self.min_cluster:
                self.used_clusters.add(k)
        # TODO: Fix it so that proximal distance is only used when relevant
        # if a cluster has a mean within a certain distance, then create a narrowed cluster
        self.key_mapping = dict()
        k = 0
        print(mode_models.mean())
        for i, mean in enumerate(mode_models.mean()[0]):
            # print(mean)
            if np.sum(np.abs(mean)) < self.prox_distance and i in self.used_clusters:
                self.key_mapping[i] = k
                k += 1
            elif i in self.used_clusters:
                self.key_mapping[i] = -1
        self.num_mappings = k
        print(self.key_mapping, self.num_mappings)

    def collapse_assignments(self, assigned_modes):
        mode_assignments = []
        for am in assigned_modes:
            if am[0] not in self.used_clusters:
                mode_assignments.append(-1)
            else:
                mode_assignments.append(self.key_mapping[am[0]])
        return np.array(mode_assignments)

class ProximityBehaviorDeterminer(ChangepointDeterminer):
    '''
    finds modes based on whether they are proximal and then the subsequent behavior (velocity)
    '''
    def __init__(self, prox_distance=3, min_cluster = 10):
        self.prox_distance = prox_distance # TODO: Set proximal distance based on ...
        self.min_cluster = min_cluster

    def fit_narrow_modes(self, changepoint_models, mode_models, assigned_modes):
        # find clusters with used 
        seen_modes = Counter()
        for mode in assigned_modes:
            seen_modes[(mode[0], mode[1])] += 1
        self.used_clusters = set()
        for k in seen_modes.keys():
            if seen_modes[k] > self.min_cluster:
                self.used_clusters.add(k)
        print(self.used_clusters)
        # TODO: Fix it so that proximal distance is only used when relevant
        self.key_mapping = dict()
        k = 0 # no mixed clusters in this case, because we only concern with proximal
        for i, mean in enumerate(mode_models.mean()[0]):
            if np.sum(np.abs(mean)) < self.prox_distance:
                for j in range(len(mode_models.mean()[1])):
                    print(mean, mode_models.mean()[1][j], seen_modes[(i,j)])
                    if (i,j) in self.used_clusters:
                        self.key_mapping[(i,j)] = k
                        k += 1
        self.num_mappings = k
        print("num mappings ", k)
        print(mode_models.mean())

    def collapse_assignments(self, assigned_modes):
        mode_assignments = []
        for am in assigned_modes:
            if (am[0], am[1]) not in self.key_mapping:
                mode_assignments.append(-1)
            else:
                print(am)
                mode_assignments.append(self.key_mapping[(am[0], am[1])])
        return np.array(mode_assignments)

determiners = {"overlap": PureOverlapDeterminer, "prox": ProximityDeterminer, "proxVel": ProximityBehaviorDeterminer}
