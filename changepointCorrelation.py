class ChangepointDeterminer():
	def __init__(self):
		n_components=10, max_iter=6000, dp_prior=100, covariance_type='diag', cov_prior=1e-10
		self.window
		self.transformer

	def cluster_modes(data, models):
	    # data = np.concatenate(model_parameters, axis =    0)
	    cov_prior = [cov_prior for _ in range(data.shape[1])]
	    mean_prior = [0 for _ in range(data.shape[1])]
	    mode_model = mix.BayesianGaussianMixture(n_components=self.n_components, max_iter=self.max_iter, 
	                                    weight_concentration_prior=self.dp_prior, covariance_type=self.covariance_type, 
	                                    covariance_prior=self.cov_prior, mean_prior=self.mean_prior) # uses a dirichlet process GMM to cluster
	    mode_weights = mode_model.weights_
	    mode_means = mode_model.means_
	    print(mode_weights, mode_means)
	    data_modes = mode_model.predict(data)
	    modes = []
	    for m, mode in zip(models, data_modes):
	        m.mode = mode
	        modes.append(mode)
	    return mode_model, modes


	def changepoint_statistics(self, models, changepoints, trajectory, correlate_trajectory):
		data = transformer.discover_modes(models, changepoints, trajectory, correlate_trajectory, self.window)
        
		

	def trajectory_statistics(self, models, changepoints, obj_dumps):
