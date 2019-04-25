import numpy as np

class ChangepointModels():
    '''
    combines the champ model, the mode cluster algorithm, and the narrowing algorithm and calls fitting for the mode cluster
    and the narrowing algorithms
    '''
    def __init__(self, args, changepoint_model, transforms, cluster_model, determiner):
        self.mode_model = cluster_model
        self.transforms = transforms
        self.changepoint_model = changepoint_model
        self.determiner = determiner
        self.window = args.window

    def changepoint_statistics(self, models, changepoints, trajectory, correlate_trajectory):
        '''
        fits the determiner, clusters and applys the transformers to the champ-applied data
        '''
        datas = []
        for transformer in self.transforms: # transformers must be ordered if multiple
            data = transformer.mode_statistics(models, changepoints, trajectory, correlate_trajectory, self.window)
            # print(data.shape)

            datas.append(data)
        # ndatas = [[] for _ in range(len(datas))]
        # for d in zip(*datas): # hardcoded outlier removal. TODO: make not hardcoded
        #     keep = True
        #     for i in range(len(d)):
        #         v = d[i]
        #         if np.sum(np.abs(v)) > 4:
        #             keep = False
        #         if keep:
        #             ndatas[i].append(v)
        # for i in range(len(self.transforms)):
        #     ndatas[i] = np.array(ndatas[i])
        self.mode_model.fit(datas)
        # print(datas)
        assignments = self.mode_model.predict(datas)
        for a, value in zip(assignments, datas[0]):
            print(a, value)
        self.determiner.fit_narrow_modes(models, self.mode_model, assignments)

    def get_mode(self, trajectory, saliency_trajectory, models=None, changepoints=None):
        '''
        given a trajectory and a saliency trajectory, gets the option mode assignments for them, by applying
        CHAMP, transforming, assigning modes, and reducing the modes to option-relevant modes
        trajectory of the form: [num_data, size of data]
        saliency_trajectory of the form: [num_data, size of data]
        returns assignments [num_changepoints, mode assignments]
        '''
        if models is None:
            models, changepoints = self.changepoint_model.generate_changepoints(trajectory)
        changepoints = changepoints.tolist()
        datas = []
        for transform in self.transforms:
            data = transform.mode_statistics(models, changepoints, trajectory, saliency_trajectory, window=self.window)
            datas.append(data)
        mode_assignments = self.mode_model.predict(datas)
        # print(self.mode_model.mean())
        # print("datas", mode_assignments, changepoints)
        mode_assignments = np.stack(mode_assignments, axis=0)
        assignments = self.determiner.collapse_assignments(mode_assignments)
        # for i in range(200):
        #     print(assignments[i], mode_assignments[i,0],datas[0][i], trajectory[changepoints[i]:changepoints[i+1]+1], changepoints[i], changepoints[i+1])
        return assignments, changepoints
# define segments/windows into single fixed length statistics
# cluster statistics into modes
# group modes based on "reproducibility"
# give each group of modes a reward function

# forms of reproducibility:
    # like behaviors
    # behavior overlapped with consistent actions
    # behavior overlapped with proximity
    # behavior overlapped with saliency

# generating changepoints:
# data of the form: [{object: values}]
# create CHAMP obj
# champ.generate_changepoints
# create transformer object
# transform changepoints
# fit clusters
# assign changepoint windows/segments to cluster model
# create changepoint mode clusters