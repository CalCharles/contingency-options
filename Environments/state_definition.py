import numpy as np

def midpoint_separation(self_other, clip_distance = 100.0, norm_distance=2.0): # no norm distance right now...
    # self_location, other_location = self_other
    self_midpoint, other_midpoint = self_other
    if sum(self_midpoint) < 0 or sum(other_midpoint) < 0:
        return [norm_distance, norm_distance] # use 10 to denote nonexistent relative values
    # self_midpoint = ((self_location[0] + self_location[2])/2.0, (self_location[3] + self_location[1])/2.0)
    # other_midpoint = ((other_location[0] + other_location[2])/2.0, (other_location[3] + other_location[1])/2.0)
    s1, s2 = np.sign(self_midpoint[0] - other_midpoint[0]), np.sign(self_midpoint[1] - other_midpoint[1])
    d1, d2 = np.abs(self_midpoint[0] - other_midpoint[0]), np.abs(self_midpoint[1] - other_midpoint[1])
    coordinate_distance = [s1 * min(d1, clip_distance), s2 * min(d2, clip_distance)]
    # print(coordinate_distance)
    return coordinate_distance

def get_proximal(correlate, data, clip=5, norm=5):
    return np.array(list(map(lambda x: midpoint_separation(x, clip_distance=clip, norm_distance=norm), zip(correlate, data))))

class Relationship():
	def compute_comparison(self, state, target, correlate):
		'''
		returns the comparison the target values and the self values, which are as full state (raw, factored) 
		'''
		pass

class Proximity(): # prox
	def compute_comparison(self, state, target, correlate):
		return midpoint_separation((state[1][target][0], state[1][correlate][0]))

class Full(): # full
	def compute_comparison(self, state, target, correlate):
		return list(state[1][correlate][0]) + list(state[1][correlate][1])

class Bounds(): # bounds
	def compute_comparison(self, state, target, correlate):
		return list(state[1][correlate][0])

class Feature(): # feature
	def compute_comparison(self, state, target, correlate):
		return list(state[1][correlate][1])

class Raw(): # raw
	def compute_comparison(self, state, target, correlate):
		return state[0]

class Sub(): # sub # TODO: implement
	def compute_comparison(self, state, target, correlate):
		return 



class StateGet():
	'''
	State in this context comes in two forms:
		dictionary of state names to object locations and properties (factor_state)
		the image represented state (raw_state)

	'''
	def __init__(self, action_num, target):
		# TODO: full and feature is set at 1, and prox and bounds at 2, but this can differ
		# self.state_shapes = state_shapes
		global state_functions, state_shapes
		self.state_functions = state_functions
		self.state_shape = state_shapes
		self.action_num = action_num
		self.target = target

	def get_state(self, state):
		''' 
		state is as defined in the environment class, that is, a tuple of
			(raw_state, factor_state)
		'''
		pass

	def get_minmax(self):
		return self.minmax

	def set_minmax(self, minmax):
		self.minmax = minmax

class GetState(StateGet):
	'''
	gets a state with components as defined above
	'''
	def __init__(self, action_num, state_forms, target):
		'''
		given a list of pairs (name of correlate, relationship)
		'''
		super(GetState, self).__init__(action_num, target)
		# TODO: does not work on combination of higher dimensions
		# TODO: order of input matters/ must be fixed
		self.shape = np.sum([self.state_shape[state_form[1]] for state_form in state_forms], axis=0)
		self.names = [state_form[0] for state_form in state_forms]
		self.functions = [self.state_functions[state_form[1]] for state_form in state_forms]
		self.action_num=action_num

	def get_state(self, state):
		state = []
		for f, (name, form) in zip(self.names, self.functions):
			state += f(state, self.target, name)
		return np.array(state)

state_functions = {"prox": Proximity(), "full": Full(), "bounds": Bounds(), 
							"feature": Feature(), "raw": Raw(), "sub": Sub()}
# TODO: full and feature is currently set at 1, and prox and bounds at 2, but this can differ
state_shapes = {"prox": [2], "full": [3], "bounds": [2], "feature": [1], "raw": [64, 64], "sub": [4,4]}
# class GetRaw(StateGet):
# 	'''
# 	Returns the raw_state
# 	'''
#     def __init__(self, state_shape, action_num, minmax, correlate_size):
#         super(GetProximal, self).__init__(state_shape, action_num, minmax, correlate_size)

#     def get_state(self, state):
#         return state[0]

# class GetFactored(StateGet):
# 	def __init__(self, state_shape, action_num, minmax, correlate_size, target_name="", correlate_names=[""]):
# 		super(GetFactored, self).__init__(state_shape, action_num, minmax, correlate_size)
# 		self.target_name = target_name
# 		self.correlates = correlate_names

#     def get_state(self, state):
#         return state[self.target_name][1]

# class GetBoundingBox(StateGet):
# 	def __init__(self, state_shape, action_num, minmax, correlate_size, target_name="", correlate_names=[""]):
# 		super(GetBoundingBox, self).__init__(state_shape, action_num, minmax, correlate_size)
# 		self.target_name = target_name
# 		self.correlates = correlate_names

#     def get_state(self, state):
#         return state[self.target_name][1][0]

# class GetProperties(StateGet):
# 	def __init__(self, state_shape, action_num, minmax, correlate_size, target_name="", correlate_names=[""]):
# 		super(GetProperties, self).__init__(state_shape, action_num, minmax, correlate_size)
# 		self.target_name = target_name
# 		self.correlates = correlate_names

#     def get_state(self, state):
#         return state[self.target_name][1][1]

# class GetProximal(StateGet):
# 	def __init__(self, state_shape, action_num, minmax, correlate_size, target_name="", correlate_names=[""]):
# 		super(GetProperties, self).__init__(state_shape, action_num, minmax, correlate_size)
# 		self.target_name = target_name
# 		self.correlates = correlate_names

    # def get_state(self, state):
    # 	prox = midpoint_separation((state[self.target_name][1][0], state[self.correlate_names[0]][1][0]))
    #     return np.concatenate([state[self.target_name][1][0], state[self.target_name][1][0]], axis=1)
