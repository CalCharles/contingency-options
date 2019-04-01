class Kernel(nn.Module):
    def __init__(self, args, num_inputs):
    	'''
		num inputs is the number of inputs that a single of the two inputs is over
    	'''
    	pass

    def forward(self, x, y):
    	'''
		R^n x R^n -> R
    	'''
    	pass

class GaussianKernel(Kernel):
	def __init__(self, args, num_inputs):
		self.num_inputs = num_inputs
		self.h = 1

	def forward(self, x, y):
		return torch.exp(-(x - y).pow(2).sum() / self.h)

kernels = {"gaussian": GaussianKernel}