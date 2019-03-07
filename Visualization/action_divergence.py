

def compute_KL(data, first_policy, second_policy):
	values1, dist_entropy1, action_probs1, q_values1 = first_policy.determine_action(data)
	values2, dist_entropy2, action_probs2, q_values2 = second_policy.determine_action(data)
    values1, action_probs1, q_values1 = first_policy.get_action(values1, action_probs1, q_values1)
    values2, action_probs2, q_values2 = second_policy.get_action(values2, action_probs2, q_values2)
	# [batch, num actions]
	total_KL_divergence = (action_probs1 * torch.log(action_probs1 / action_probs2)).sum(dim=1)
	# max_KL_divergence = (action_probs1 * torch.log(action_probs1 / action_probs2)).sum(dim=1).max()
	# average_KL_divergence = (action_probs1 * torch.log(action_probs1 / action_probs2)).sum(dim=1).mean()
	return total_KL_divergence

def compute_trust_region(data, policies, epsilon=.1):
	'''
	returns the policies within epsilon of the last policy, and sequential
	'''
	policies.reverse()
	for i in range(len(policies)):
		tKL = compute_KL(data, policies[i], policies[0])
		dist = tKL.max()
		# dist = tKL.mean()
		if dist > epsilon:
			break
	policy_range = policies[:i]
	policy_range.reverse()
	return policy_range

def compute_state_distribution(args, data):
	'''
	fit a gaussian mixture model to the data, and returns data probability
	'''
    cov_prior = [args.dp_gmm[4] for _ in range(data.shape[1])]
    # mean_prior = [self.dp_gmm[5] for _ in range(data.shape[1])]
    mean_prior = [0 for _ in range(data.shape[1])]
    model = mix.BayesianGaussianMixture(n_components=args.dp_gmm[0], max_iter=args.dp_gmm[1], 
                                weight_concentration_prior=args.dp_gmm[2], covariance_type=args.dp_gmm[3], 
                                covariance_prior=cov_prior, mean_prior=mean_prior) # uses a dirichlet process GMM to cluster
 	model.fit(data)
 	probs = model.predict_proba(data)
 	return model, probs