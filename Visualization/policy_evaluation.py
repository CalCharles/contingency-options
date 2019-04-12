
# forms of policy degeneration:
	# random or fixed output
	# input agnosticism
	# reducing expectation of reward

class dist_fitter(): # TODO: merge with ClusterModel in changepointClusterModels
	def __init__(self, args, data):
		pass

	def fit(self, data):
		pass

	def predict(self, data):
		pass

class GMM_fitter():
	def __init__(self,args,data):
		self.cov_prior = data.var(dim=0)
	    self.mean_prior = data.mean(dim=0)
	    self.model = mix.BayesianGaussianMixture(n_components=args.dp_gmm[0], max_iter=args.dp_gmm[1], 
	                    weight_concentration_prior=args.dp_gmm[2], covariance_type=args.dp_gmm[3], 
	                    covariance_prior=self.cov_prior, mean_prior=self.mean_prior) # uses a dirichlet process GMM to cluster
	def fit(self, data):
	    self.model.fit(data)

	def probs(self, data):
		return self.model.predict_prob(data)

def compute_entropy(dist):
	return -((dist) * torch.log(dist + 1e-15))

def basic_degeneration(policy, data, args, dist_class):
	'''
	measures the entropy of the distribution. If the batch entropy (measure of unique actions) is low, this is
	a fixed output policy. Similarly, if the dist entropy (probability of taking an action in a particular state)
	is high, this is a random output policy 
	batch_qent is the assignment of maximal q values
	dist_q is the distribution of q values, modeled by: first and second moments, GMM
	'''
	values, dist_entropy, action_probs, q_values = policy.determine_action(data)
    values, action_probs, q_values = train_models.get_action(values, action_probs, q_values)
    dist_ent = -(action_probs.squeeze() * torch.log(action_probs.squeeze() + 1e-15)).sum(dim=1).mean()
    batch_mean = action_probs.squeeze().mean(dim=0)
    batch_ent = -((batch_mean) * torch.log(batch_mean + 1e-15)).sum()
    greedy_q = q_values.squeeze().max(dim=1)[0]
    batch_qent = torch.zeros(q_values.shape)
    batch_qent[list(range(len(q_values))), q_values] = 1
    batch_qent = batch_qent.mean(dim=0)
    batch_qent = -batch_qent * torch.log(batch_qent + 1e-15)
    qmodel = dist_class(args, data)
    qmodel.fit(data)
    dist_q = (q_values.mean(dim=0), q_values.var(dim=0), qmodel)
    return dist_ent, batch_ent, dist_q, batch_qent

def input_correlation(data, policy, args, dist_class):
	'''
	Measures level of correlation between inputs and outputs. This is returned as mutual information
	approximates the joint distribution, and q value distribution with a GMM
	returns correlation between action probabilities, q distribution, max q distribution, 
	'''
	values, dist_entropy, action_probs, q_values = policy.determine_action(data)
    values, action_probs, q_values = train_models.get_action(values, action_probs, q_values)
    pxm = dist_class(args,data)
    pxm.fit(data) # model of the input distribution
    px = pxym.probs(data)
    pxy = px * action_probs # probability of each action at each state
    py = action_probs.squeeze().mean(dim=0)
    action_prob_state = (pxy * torch.log(pxy/(px * py + 1e-15))).sum()

    greedy_q = q_values.squeeze().max(dim=1)[0]
    pqg = torch.zeros(q_values.shape) # pqg: prob q greedy
    pqg[list(range(len(q_values))), q_values] = 1
    pqg = pqg.mean(dim=0)
    pqgx = pqg * action_probs
    qgreedy_prob_state = (pqgx * torch.log(pqgx/(px * pqg + 1e-15))).sum()

    qx = torch.cat((data, q_values), dim=1)
    pqxm = dist_class(args,torch.cat((data, q_values), dim=0))
    pqxm.fit(qx) # model of the input distribution
    pqx = pqxm.probs(data)
    pqm = dist_class(args,q_values)
    pqm.fit(q_values) # model of the input distribution
    pq = pqm.probs(q_values)
    q_prob_state = (pqx * torch.log(pqx/(px * pq + 1e-15))).sum()
    return action_prob_state, qgreedy_prob_state, q_prob_state

def return_estimation(reward, returns, actions, eval_states, policy, hist, conf):
	''' 
	given rollouts, search for locations in the buffer with high returns. 
	Determine if similar actions were taken
	'''
	idxes = reward > 0.0
	all_compare_actions, all_compare_qvals = [], []
	for i in range(1,hist+1):
		cidx = idxes-i
		values, dist_entropy, action_probs, q_values = policy.determine_action(eval_states[cidx])
	    values, action_probs, q_values = train_models.get_action(values, action_probs, q_values)
	    compare_actions = torch.clamp(np.pow(conf, i) - action_probs[actions[cidx]], 0, 1)
		compare_qvalues = q_values[cidx] - returns
		all_compare_actions.append(compare_actions)
		all_compare_qvals.append(compare_qvals)
	action_diffs, qval_diffs = list(zip(*all_compare_actions)), list(zip(*all_compare_qvals))
	return idxes, torch.stack(action_diffs), torch.stack(qval_diffs)

# def true_qvalue_evaluation(proxy_environment, ):


# def return_evaluation():
# 	'''
# 	get the actual full returns of evaluating a policy
# 	'''
