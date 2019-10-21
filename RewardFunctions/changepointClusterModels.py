import sklearn as sk
import sklearn.mixture as mix
import numpy as np

class ClusterModel():
    def ___init__(self):
        pass

    def fit(self, datas):
        pass

    def predict(self, datas):
        pass

    def mean(self):
        pass

class MultipleCluster(ClusterModel):
    def __init__(self, args, base_model):
        self.cluster_models = []
        self.args = args
        self.base_model = base_model

    def __str__(self):
        out = ""
        for i, models in enumerate(self.cluster_models):
            out += "==========Model layer"+str(i) + "==========\n"
            for m in models:
                out += str(m) + "\n"
                out += str(m.means_) + "\n"
                out += str(m.weights_)
        return out

    def fit(self, datas):
        # datas of the form ([num changepoints, transformed dim], ... )
        for data in datas:
            model = self.base_model(self.args)
            model.fit(data)
            # print(model.mean())
            self.cluster_models.append(model)

    def predict(self, datas):
        # datas of the form ([num changepoints, transformed dim], ... )
        assignments = []
        for i, data in enumerate(datas):
            assignments.append(self.cluster_models[i].predict(data))
        assignments = np.stack(assignments, axis = 1)
        return assignments

    def mean(self):
        # returns the means of all cluster models
        return [cm.mean() for cm in self.cluster_models]

class BayesianGaussianMixture(ClusterModel):
    def __init__(self, args):
        self.dp_gmm = args.dp_gmm
        self.model = None

    def fit(self, data):
        cov_prior = [self.dp_gmm[4] for _ in range(data.shape[1])]
        # mean_prior = [self.dp_gmm[5] for _ in range(data.shape[1])]
        mmin, mmax = np.min(data, axis=0), np.max(data, axis=0)
        rng = mmax - mmin
        mmean = np.mean(data, axis=0)
        mean_prior = [0 for i in range(data.shape[1])]
        # mean_prior = [mmin + (rng/data.shape[1] * i) for i in range(data.shape[1])]
        # print(mmin, mmax, mean_prior)
        # error
        self.model = mix.BayesianGaussianMixture(n_components=self.dp_gmm[0], max_iter=self.dp_gmm[1], 
                                        weight_concentration_prior=self.dp_gmm[2], covariance_type=self.dp_gmm[3], 
                                        covariance_prior=cov_prior, mean_prior=mean_prior) # uses a dirichlet process GMM to cluster
        return self.model.fit(data)

    def predict(self, data):
        return self.model.predict(data)

    def mean(self):
        return self.model.means_

class FilteredBayesianGaussianMixture(ClusterModel):
    def __init__(self, args):
        self.dp_gmm = args.dp_gmm
        self.args = args
        self.model = None
        self.prox = args.period

    def fit(self, data):
        self.model = BayesianGaussianMixture(self.args)
        # fits = np.zeros(data.shape[0])
        print(data[data[:,-1] <= self.prox].shape)
        use_data = data[data[:,-1] <= self.prox]
        use_data = use_data[np.sum(np.abs(use_data[:,:-1]), axis=1) <= self.prox]
        use_data = use_data[np.sum(np.abs(use_data[:,:-1]), axis=1) != 0]
        print(use_data)
        modelfit = self.model.fit(use_data[:,:-1])
        fits = modelfit.predict(data[:,:-1])
        fits[data[:,-1] > self.prox] = -1
        fits[np.sum(np.abs(data[:,:-1]), axis=1) > self.prox] = -1
        fits[np.sum(np.abs(data[:,:-1]), axis=1) == 0] = -1
        print(fits.shape)
        return self

    def predict(self, data):
        fits = self.model.predict(data[:,:-1])
        fits[data[:,-1] > self.prox] = -1
        fits[np.sum(np.abs(data[:,:-1]), axis=1) > self.prox] = -1
        fits[np.sum(np.abs(data[:,:-1]), axis=1) == 0] = -1
        # print(fits.shape)
        return fits

    def mean(self):
        return self.model.model.means_

cluster_models = {"DPGMM": BayesianGaussianMixture, "FDPGMM": FilteredBayesianGaussianMixture}