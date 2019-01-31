class EpsilonGreedyQ():

    def initialize(self, args, num_outputs):
        self.epsilon = args.greedy_epsilon
        self.num_outputs = num_outputs

    def take_action(self, probs, q_vals):
        action = sample_actions(F.softmax(q_vals, dim=1), deterministic =True)
        if np.random.rand() < .1:
            action = pytorch_model.wrap(np.random.randint(self.num_outputs, size = q_vals.shape[0]), cuda = True)
        return action

class EpsilonGreedyProbs():

    def initialize(self, args, num_outputs):
        self.epsilon = args.greedy_epsilon
        self.num_outputs = num_outputs

    def take_action(self, probs, q_vals):
        action = sample_actions(probs, deterministic =True)
        if np.random.rand() < .1:
            action = pytorch_model.wrap(np.random.randint(self.num_outputs, size = probs.shape[0]), cuda = True)
        return action
