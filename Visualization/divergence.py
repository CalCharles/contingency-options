import sklearn as sk
import sklearn.mixture as mix
import numpy as np

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

def compute_l2Q(data, first_policy, second_policy)
	values1, dist_entropy1, action_probs1, q_values1 = first_policy.determine_action(data)
	values2, dist_entropy2, action_probs2, q_values2 = second_policy.determine_action(data)
    values1, action_probs1, q_values1 = first_policy.get_action(values1, action_probs1, q_values1)
    values2, action_probs2, q_values2 = second_policy.get_action(values2, action_probs2, q_values2)
    total_L2 = (q_values1 - q_values2).abs().sum(dim=1)
    return total_L2

def compute_relQ(data, first_policy, second_policy):
	'''
	computes number of the same highest Q values (alternatively, ranking order)
	'''
	values1, dist_entropy1, action_probs1, q_values1 = first_policy.determine_action(data)
	values2, dist_entropy2, action_probs2, q_values2 = second_policy.determine_action(data)
    values1, action_probs1, q_values1 = first_policy.get_action(values1, action_probs1, q_values1)
    values2, action_probs2, q_values2 = second_policy.get_action(values2, action_probs2, q_values2)
    return torch.eq(q_values1.max(dim = 1)[1], q_values2.max(dim = 1)[1]).float()


# def compute_Q_dist

def compute_trust_region(data, policies, dist_metric, epsilon=.1):
	'''
	returns the policies within epsilon of the last policy, and sequential
	'''
	policies.reverse()
	for i in range(len(policies)):
		div = dist_metric(data, policies[i], policies[0])
		dist = div.max()
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

def compute_weighted_div(data, first_policy, second_policy, probs, dist_metric):
	return dist_metric(data, first_policy, second_policy) * probs # variational distance ignoring 
