import numpy as np
import imageio as imio
from file_management import read_obj_dumps
import os

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
		returns flattened partial state
		'''
		pass

class Velocity(): # prox
	def __init__(self):
		self.lastpos = None

	def compute_comparison(self, state, target, correlate):
		if self.lastpos is None:
			self.lastpos = np.array(state[1][correlate][0])
		rval= (np.array(state[1][correlate][0]) - self.lastpos).tolist()
		self.lastpos = np.array(state[1][correlate][0])
		return rval

class Acceleration(): # prox
	def __init__(self):
		self.llpos = None
		self.lastpos = None

	def compute_comparison(self, state, target, correlate):
		if self.lastpos is None:
			self.lastpos = np.array(state[1][correlate][0])
			self.llpos = np.array(state[1][correlate][0])
		rval= ((np.array(state[1][correlate][0]) - self.lastpos) - (self.lastpos - self.llpos)).tolist()
		self.llpos = self.lastpos
		self.lastpos = np.array(state[1][correlate][0])
		return rval

class BinaryExistence():
	def compute_comparison(self, state, target, correlate):
		obj_dump = state[1]
		names = list(obj_dump.keys())
		names.sort()
		states = []
		for name in names:
			if name.find(correlate) != -1:
				if obj_dump[name][1]:
					states += obj_dump[name][1]
		return states


class Proximity(): # prox
	def compute_comparison(self, state, target, correlate):
		return midpoint_separation((np.array(state[1][target][0]), np.array(state[1][correlate][0])))

class MultiFull(): # multi-object bounds
	def compute_comparison(self, state, target, correlate):
		states = []
		obj_dump = state[1]
		for name in obj_dump.keys():
			if name.find(correlate) != -1:
				states += list(obj_dump[name][0]) + list(obj_dump[name][1])
		return states

class Full(): # full
	def compute_comparison(self, state, target, correlate):
		return list(state[1][correlate][0]) + list(state[1][correlate][1])

class Bounds(): # bounds
	def compute_comparison(self, state, target, correlate):
		return list(state[1][correlate][0])

class MultiVisibleBounds(): # multi-object bounds
	def compute_comparison(self, state, target, correlate):
		states = []
		obj_dump = state[1]
		for name in obj_dump.keys():
			if name.find(correlate) != -1:
				if obj_dump[name][1]:
					states += obj_dump[name][0]
		return states

class XProximity(): # bounds
	def compute_comparison(self, state, target, correlate):
		# print(midpoint_separation((np.array(state[1][target][0]), np.array(state[1][correlate][0])))[1])
		return list([midpoint_separation((np.array(state[1][target][0]), np.array(state[1][correlate][0])))[1]])

class Feature(): # feature
	def compute_comparison(self, state, target, correlate):
		return list(state[1][correlate][1])

class Raw(): # raw
	def compute_comparison(self, state, target, correlate):
		return state[0].flatten().tolist()

class Sub(): # sub # TODO: implement
	def compute_comparison(self, state, target, correlate):
		return 



class StateGet():
	'''
	State in this context comes in two forms:
		dictionary of state names to object locations and properties (factor_state)
		the image represented state (raw_state)

	'''
	def __init__(self, target, minmax):
		# TODO: full and feature is set at 1, and prox and bounds at 2, but this can differ
		# self.state_shapes = state_shapes
		global state_functions, state_shapes
		self.state_functions = state_functions
		self.state_shape = state_shapes
		self.target = target
		self.minmax = minmax
		self.shape = None # should always be defined at some point

	def get_state(self, state):
		''' 
		state is as defined in the environment class, that is, a tuple of
			(raw_state, factor_state)
		returns: raw_state, responsibility (number of values associated with input pair)
		'''
		pass

	def flat_state_size(self):
		return int(np.prod(self.shape))

	def get_minmax(self):
		return self.minmax

	def set_minmax(self, minmax):
		self.minmax = minmax

class GetState(StateGet):
	'''
	gets a state with components as defined above
	'''
	def __init__(self, target, minmax=None, state_forms=None):
		'''
		given a list of pairs (name of correlate, relationship)
		'''
		super(GetState, self).__init__(target, minmax=minmax)
		# TODO: does not work on combination of higher dimensions
		# TODO: order of input matters/ must be fixed
		self.shape = np.sum([self.state_shape[state_form[1]] for state_form in state_forms])
		self.shapes = {(state_form[0], state_form[1]): self.state_shape[state_form[1]] for state_form in state_forms} # dimensionality for each component
		self.fnames = [state_form[1] for state_form in state_forms]
		self.names = [state_form[0] for state_form in state_forms]
		self.name = "-".join([s[0] for s in state_forms] + [s[1] for s in state_forms])
		self.functions = [self.state_functions[state_form[1]] for state_form in state_forms]

	def get_state(self, state):
		estate = []
		resp = []
		for name, f in zip(self.names, self.functions):
			comp = f.compute_comparison(state, self.target, name)
			resp.append(len(comp))
			estate += comp
		return np.array(estate), resp 

	def determine_delta_target(self, states):
		'''
		given a set of values, determine if any of them changed. assumes target is the first resp
		returns index of difference (assumes only 1), and index in the state of difference (assumes only 1)
		'''
		last_shape = self.shapes[(self.names[0], self.fnames[0])][0] # assumes 1D
		change_indexes,ats,rstates = [], [], []
		for i, (s1, s2) in enumerate(zip(states[:-1], states[1:])):
			diff = s1[:last_shape] - s2[:last_shape]
			mag = np.linalg.norm(diff)
			if mag > 0:
				lidx = np.where(diff != 0)[0][0]
				ats.append((lidx + 1) // 3) # 2 and 3 hard coded as the x,y,attribute
				rstates.append(s1[:last_shape][lidx-2:lidx])
				change_indexes.append(i+1)
		return change_indexes, ats, rstates

	def determine_target(self, states, resps):
		'''
		given a set of states, return the locations of the states for which the target disappeared
		assumes only one change index, returns the first, -1 if no index
		returns change index and changed state
		'''
		last_resp = np.sum(resps[0])
		change_index = -1
		for i, resp in enumerate(resps[1:]):
			r = np.sum(resp)
			if last_resp != r:
				change_index = i + 1
				break
			last_resp = r
		return_state = None
		if change_index != -1:
			r1, r2 = resps[change_index - 1], resps[change_index]
			s1, s2 = states[change_index - 1], states[change_index]
			rat = 0
			for i, (rv1, rv2) in enumerate(zip(r1, r2)):
				if rv1 != rv2: # assumes only one component of state has changed
					s1, s2 = s1[rat:rat + rv1], s2[rat:rat+rv2]
					break
				else:
					rat += rv1
			shpe = self.shapes[(self.names[i], self.fnames[i])][0] # assumes 1D shape
			s1, s2 = s1.reshape(len(sl) // shpe, shpe), s2.reshape(len(s2) // shpe, shpe) # hopefully reshape does as desired
			l1, l2 = s1.shape[0], s2.shape[0]
			sm1, sm2 = np.stack([s1 for s1 in s2.shape[0]]), np.stack([s2 for s2 in s1.shape[0]])
			diff_mat = np.linalg.norm(sm1 - sm2.T, axis=2)
			if l1 > l2: # an object disappeared
				closest = np.argmin(np.min(diff_mat, axis=1))
				return_state = s1[closest]
			else: # an object appeared, there should never be l1 == l2
				closest = np.argmin(np.min(diff_mat, axis=0))
				return_state = s2[closest]
		return change_index, closest, return_state


class GetRaw(StateGet):
	'''
	gets a state with components as defined above
	'''
	def __init__(self, target="", minmax=None, state_forms=None, state_shape = None):
		'''
		given a list of pairs (name of correlate, relationship)
		'''
		super(GetRaw, self).__init__(target, minmax=minmax)
		self.shape = np.sum(state_shape)		

	def get_state(self, state):
		raw = state[0].flatten()
		return raw, [len(raw)]

def load_states(state_function, pth, length_constraint=50000, use_raw = False, raws = None, dumps = None):
	if raws is None:
		raw_files = []
		if use_raw:
			for root, dirs, files in os.walk(pth, topdown=False):
				dirs.sort(key=lambda x: int(x))
				print(pth, dirs)
				for d in dirs:
					try:
						for p in [os.path.join(pth, d, "state" + str(i) + ".png") for i in range(2000)]:
							raw_files.append(imio.imread(p))
							if len(raw_files) > length_constraint:
								raw_files.pop(0)
					except OSError as e:
						# reached the end of the file
						pass
	else:
		raw_files = raws
	if dumps is None:
		dumps = read_obj_dumps(pth, i=-1, rng = length_constraint)
	else:
		dumps = dumps
	print(len(raw_files), len(dumps))
	if len(raw_files) < len(dumps) and not use_raw:
		# raw files not saved for some reason, which means use a dummy array of the same length
		raw_files = list(range(len(dumps)))
	states = []
	resps = []
	for state in zip(raw_files, dumps):
		sv, resp = state_function(state)
		states.append(sv)
		resps.append(np.array(resp))
	states = np.stack(states, axis=0)
	resps = np.stack(resps, axis=0)
	return states, resps, raw_files, dumps


def compute_minmax(state_function, pth):
	'''
	assumes pth leads to folder containing folders with raw images, and object_dumps file
	uses the last 50000 data points, or less
	'''
	saved_minmax_pth = os.path.join(pth, state_function.name + "_minmax.npy")
	print(saved_minmax_pth)
	try:
		print("loaded minmax from: ", saved_minmax_pth)
		minmax = np.load(saved_minmax_pth)
	except FileNotFoundError as e:
		print("not loaded", saved_minmax_pth)
		use_raw = 'raw' in state_function.names
		states, resps, raws, dumps = load_states(state_function.get_state, pth, use_raw = use_raw) # TODO: no normalization for raw states (not implemented)
		minmax = (np.min(states, axis=0), np.max(states, axis=0))
		np.save(saved_minmax_pth, minmax)
	print(minmax)
	return minmax




state_functions = {"prox": Proximity(), "full": Full(), "bounds": Bounds(), 'vismulti': MultiVisibleBounds(), 
					"vel": Velocity(), "acc": Acceleration(), "xprox": XProximity(), 'bin': BinaryExistence(),
					"feature": Feature(), "raw": Raw(), "sub": Sub(), "multifull": MultiFull()}
# TODO: full and feature is currently set at 1, and prox and bounds at 2, but this can differ, bin has hardcoded size, as does multifull
state_shapes = {"prox": [2], "xprox": [1], "full": [3], "bounds": [2], "vel": [2], "acc": [2], 'bin': [100], "multifull": [300], "feature": [1], "raw": [64, 64], "sub": [4,4]}
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
