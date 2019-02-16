import numpy as np

from ChangepointDetection.ChangepointDetectorBase import ChangepointDetector


EPS = 1e-8


# simple linear CPD, detects based on differences in direction
class LinearCPD(ChangepointDetector):
	def __init__(self, threshold_angle, *args, **kwargs):
		self.threshold = np.cos(threshold_angle)

	def generate_changepoints(self, data):
		d_diff = data[1:] - data[:-1]
		d_diff_nm = d_diff / (np.linalg.norm(d_diff, axis=1).reshape(-1, 1)+EPS)
		ang_diff = np.multiply(d_diff_nm[1:], d_diff_nm[:-1]).sum(axis=1)
		cp_mask = np.logical_and(
			ang_diff < self.threshold,
			np.linalg.norm(d_diff, axis=1)[1:] > EPS)
		changepoints = np.where(cp_mask)[0] + 1  # shift forward
		return changepoints