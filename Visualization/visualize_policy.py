

class StateVisualizer():
	def __init__(self, minmax, widths):

		self.minmax = minmax
        for minv, maxv in zip(minvs, maxvs):
            order_vector = []
            for i in range (self.order):
                if not self.normalize:
                    order_vector.append((minv + i * (maxv - minv) / (self.order - 1)))
                else:
                    order_vector.append((i / (self.order - 1)))
            self.order_vectors.append(pytorch_model.wrap(np.array(order_vector)))

	def visualize_policy(models, value_set):
		for model in models.models:
		  pass