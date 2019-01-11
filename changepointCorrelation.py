def overlap_clusters(modes, mode_models):
    '''
    Given two sources of information (mode_models), overlap and decide on which clusters are indicative
    '''
    mode_corresp = {i:[] for i in range(10)}
    for m1, m2 in zip(modes[0], modes[1]):
        mode_corresp[m1].append(m2)
    print(mode_models[0].weights_, mode_models[0].means_)

    print(mode_corresp)

    # compute overlap of different modes:
    print([len(mode_corresp[k]) / len(modes[0]) for k in mode_corresp.keys()])
    keys = [k for k in mode_corresp.keys() if len(mode_corresp[k]) / len(modes[0]) > .05]
    discounted_keys = [k for k in mode_corresp.keys() if len(mode_corresp[k]) / len(modes[0]) <= .05]
    overlap_correspondence = dict()
    oscore = collections.Counter()
    val_counts = collections.Counter()
    olap_values = []
    for key in keys:
        overlap = dict()
        for val1 in mode_corresp[key]:
            val_counts[val1] += 1
            if val1 in overlap:
                overlap[val1] += 1
            else:
                overlap[val1] = 1
        olap_values.append(overlap)
    print(val_counts)
    for overlap, key in zip(olap_values, keys):
        for v in overlap.keys():
            print("counts", v, key, (len(mode_corresp[key])+ val_counts[v]))
            overlap[v] = overlap[v] / (len(mode_corresp[key])+ val_counts[v])
        overlap_correspondence[key] = overlap
        if len(list(overlap.keys())) > 0:
            oscore[key] = 0
    print("correspondence", overlap_correspondence)
    # find overlap between different keys
    overlapping = dict()
    for i in range(len(keys)):
        key1 = keys[i]
        for j in range(i, len(keys)):
            key2 = keys[j]
            if key1 != key2:
                for v in overlap_correspondence[key1].keys():
                    if v in overlap_correspondence[key2]:
                        k1, k2 = min(key1, key2), max(key1, key2)
                        if (k1, k2) in overlapping:
                            overlapping[(k1,k2)].append((v, overlap_correspondence[k1][v], overlap_correspondence[k2][v]))
                        else:
                            overlapping[(k1,k2)] = [(v, overlap_correspondence[k1][v], overlap_correspondence[k2][v])]
    # assign a score to various keys based on their overlap. Favor more distinct keys when learning?
    # current scoring: percent of self compared multiplied with percent of other
    # higher is more overlap
    print(overlapping)
    overlapping_values = collections.Counter()
    for (k1, k2) in overlapping.keys():
        for overlap in overlapping[(k1, k2)]:
            print(k1, k2, overlap)
            oscore[k1] += overlap[1] * overlap[2]
            oscore[k2] += overlap[1] * overlap[2]
            overlapping_values[(k1,k2)] += overlap[1] * overlap[2]
    print(oscore)
    return oscore, overlapping_values


class ChangepointDeterminer():
	def __init__(self):
		n_components=10, max_iter=6000, dp_prior=100, covariance_type='diag', cov_prior=1e-10
		self.window
		self.transformers
		self.mode_models
		self.CHAMP

	def cluster_modes(data, models):	
	    # data = np.concatenate(model_parameters, axis =    0)
	    cov_prior = [cov_prior for _ in range(data.shape[1])]
	    mean_prior = [0 for _ in range(data.shape[1])]
	    self.mode_model = mix.BayesianGaussianMixture(n_components=self.n_components, max_iter=self.max_iter, 
	                                    weight_concentration_prior=self.dp_prior, covariance_type=self.covariance_type, 
	                                    covariance_prior=self.cov_prior, mean_prior=self.mean_prior) # uses a dirichlet process GMM to cluster
	    mode_weights = self.mode_model.weights_
	    mode_means = self.mode_model.means_
	    print(mode_weights, mode_means)
	    data_modes = self.mode_model.predict(data)
	    modes = []
	    for m, mode in zip(models, data_modes):
	        m.mode = mode
	        modes.append(mode)
	    return self.mode_model, modes


	def narrow_modes(models, modes, mode_values):
	    '''
	    Narrows the modes to return a set of distinct mode criterion: given data between/at two/a changepoint,
	    determines which mode cluster it belongs to, if any. All seen modes should be accounted for
	    current: Narrow modes takes the ode set with the smallest number of modes
	    TODO: Narrow modes overlaps the modes to maximize explanation of modes
	    '''
	    mode_totals = []
	    for mode_model in modes:
	        weights = mode_model.weights_
	        total = 0
	        for weight in weights:
	            if weight > .05:
	                total += 1
	        mode_totals.append(total)
	    lowest_mode = np.argmin(mode_totals)
	    cluster_indexes = [i for i in range(10)]
	    pure_clusters = []
	    mixed_clusters = []
	    seen = {lowest_mode}
	    print("modes:", len(modes))
	    if len(modes) > 1:
	    	# TODO: replace this logic with a different form of relating two segements of data to demonstrate mutual information
	    	# currently, 
	        for i in range(len(modes)):
	            if i in seen:
	                continue
	            else:
	                seen.add(i)
	                oscore, overlapping = overlap_clusters([mode_values[lowest_mode], mode_values[i]], [modes[lowest_mode], modes[i]])
	                for ci in cluster_indexes:
	                    if ci in oscore:
	                        if oscore[ci] < .03:
	                            pure_clusters.append([ci])
	                        else:
	                            mixed_clusters.append(ci)
	                cluster_head = []
	                print("ol", overlapping)
	                for ci in mixed_clusters:
	                    cluster = []
	                    for val in mixed_clusters:
	                        k1, k2 = min(ci, val), max(ci, val)
	                        if (k1,k2) in overlapping:
	                            if overlapping[(k1, k2)] > 0:
	                                cluster.append(val)
	                    cluster_head.append(cluster)
	                print(cluster_head)
	                overlaps = [[] for k in range(len(cluster_head))]
	                for k in range(len(cluster_head)):
	                    for j in range(k+1, len(cluster_head)):
	                        if len(np.intersect1d(cluster_head[k], cluster_head[j])) > 0:
	                            overlaps[k].append(j)
	                            overlaps[j].append(k)
	                print(overlaps)
	                n_cluster_idx = []
	                unused = [k for k in range(1, len(cluster_head))]
	                while len(unused) > 0:
	                    current_cluster = []
	                    frontier = [unused.pop(0)]
	                    while len(frontier) > 0:
	                        print (frontier)
	                        val = frontier.pop(0)
	                        current_cluster.append(val)
	                        for nv in overlaps[val]:
	                            if nv not in current_cluster and nv not in frontier:
	                                frontier.append(nv)
	                                if nv in unused:
	                                    unused.remove(nv)
	                    n_cluster_idx.append(current_cluster)
	                n_cluster_head = []
	                for n_cluster in n_cluster_idx:
	                    n_cluster_head.append(np.concatenate([cluster_head[idx] for idx in n_cluster]).tolist())
	                print(n_cluster_head)
	                for k in range(len(n_cluster_head)):
	                    n_cluster_head[k] = list(set(n_cluster_head[k]))
	                cluster_head = n_cluster_head
	                mixed_clusters = cluster_head
	                print(mixed_clusters)
	                # later, remove pure clusters from mixed ones for multiple different modes
	    else:
	        usable_modes = []
	        mode_model = modes[0] 
	        weights = mode_model.weights_
	        print (weights)
	        total = 0
	        for mode_i, weight in zip(range(len(weights)), weights):
	            if weight > .05:
	                usable_modes.append(mode_i)
	        # TODO: Fix it so that proximal distance is only used when relevant
	        prox_distance = 6 # TODO: Set proximal distance based on ...
	        mi = []
	        pc = []
	        for i, mean in zip(range(len(weights)), mode_model.means_):
	            if np.sum(np.abs(mean)) < prox_distance and i in usable_modes:
	                pure_clusters.append([i])
	            elif i in usable_modes:
	                mi.append(i)
	        # mixed_clusters.append(mi) # removal of the mixed clusters when doing particulars


	    print(pure_clusters, mixed_clusters)
	    return lowest_mode, pure_clusters, mixed_clusters



	def changepoint_statistics(self, models, changepoints, trajectory, correlate_trajectory):
		assigned_modes = []
		self.mode_models = list()
		for transformer in self.transfomers: # transformers must be ordered if multiple
			data = transformer.discover_modes(models, changepoints, trajectory, correlate_trajectory, self.window)
	        model, modes = cluster_modes(data, models)
	        self.mode_models.append(model)
	        assigned_modes.append(modes)
	    lowest_mode, self.pc, self.mc = self.narrow_modes(models, self.mode_models, assigned_modes)

	def trajectory_statistics(self, obj_dumps):
		models, changepoints = self.CHAMP.generate_changepoints(trajectory)
		changepoints = np.append(changepoints, len(trajectory))
		cps = [(changepoints[i-1], changepoints[i], changepoints[i+1], models[i-1], models[i]) for i in range(1, len(changepoints)-1)]
