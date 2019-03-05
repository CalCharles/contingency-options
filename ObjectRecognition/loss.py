from __future__ import division, absolute_import, print_function
import numpy as np

import logging
logging.basicConfig(format='%(levelname)s [%(asctime)s]: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

from ChangepointDetection.LinearCPD import LinearCPD
import ObjectRecognition.util as util


# abstracted class to measure focus
class FocusLoss:

    def forward(self, focus):
        raise NotImplemented

    def __call__(self, focus):
        return self.forward(focus)


# abstracted class to measure changepoint relevance
class FocusChangePointLoss:

    def forward(self, focus, changepoints, ret_extra=False):
        raise NotImplemented

    def __call__(self, focus, changepoints, ret_extra=False):
        return self.forward(focus, changepoints, ret_extra)


# Salience Loss: award effective focus
class SaliencyLoss(FocusLoss):

    def __init__(self, frame_source, *args, **kwargs):
        self.frame_source = frame_source
        self.nb_size = kwargs.get('nb_size', (15, 20))
        self.c_fn_1 = kwargs.get('c_fn_1', util.mean_square_deviation)
        self.c_fn_2 = kwargs.get('c_fn_2', util.hinged_mean_square_deviation)

        self.frame_dev_coeff = kwargs.get('frame_dev_coeff', 1.0)
        self.focus_dev_coeff = kwargs.get('focus_dev_coeff', 1.0)
        self.frame_var_coeff = kwargs.get('frame_var_coeff', 1.0)
        self.belief_dev_coeff = kwargs.get('belief_dev_coeff', 1.0)
        self.verbose = kwargs.get('verbose', False)


    # evaluate focus, lesser = more salience
    def forward(self, focus):
        frames = util.extract_neighbor(
			self.frame_source,
			focus,
            np.arange(focus.shape[0]),
            nb_size=self.nb_size)
        frame_dev = self._deviation_1(frames)
        focus_dev = self._deviation_2(focus)
        frame_var = self._variance(frames)

        belief_dev = 0
        if self.belief_dev_coeff != 0:
            pre_frames = util.extract_neighbor(
                self.frame_source,
                focus[1:],
                np.arange(focus.shape[0] - 1),
                nb_size=self.nb_size)  # I_{t}(x_{t+1})
            post_frames = util.extract_neighbor(
                self.frame_source,
                focus[:-1],
                np.arange(focus.shape[0] - 1) + 1,
                nb_size=self.nb_size)  # I_{t+1}(x_{t})
            belief_dev = self._belief_deviation(focus, frames, 
                                                pre_frames, post_frames)

        if self.verbose:
            logger.info(
                'frame_dev= %f, focus_dev= %f, frame_var= %f, belief_dev= %f'%(
                frame_dev, focus_dev, frame_var, belief_dev))
        return  self.frame_dev_coeff*frame_dev \
            + self.focus_dev_coeff*focus_dev \
            - self.frame_var_coeff*frame_var \
            - self.belief_dev_coeff*belief_dev


    # feature deviation from consecutive elements
    def _deviation_1(self, features):
        r"""
        Sum of difference between features
            \sum_t || I(x_t) - I(x_{t+1})||^2
        where x_t = (i, j)_t
        """
        diffs = [self.c_fn_1(features[t], features[t+1]) 
                 for t in range(features.shape[0]-1)]
        return np.mean(diffs)


    # feature deviation from consecutive elements
    def _deviation_2(self, features):
        r"""
        Sum of difference between features with a hinge
            max( \sum_t || I(x_t) - I(x_{t+1})||^2 - ALPHA, 0)
        where x_t = (i, j)_t
        """
        diffs = [self.c_fn_2(features[t], features[t+1]) 
                 for t in range(features.shape[0]-1)]
        return np.mean(diffs)


    # frame non-uniformity to filter out blank frame
    def _variance(self, features):
        r"""
        metric for non-uniformity, variance
        """
        return np.mean(np.var(features.reshape((features.shape[0], -1)), axis=1))


    # belief change by frame deviation
    def _belief_deviation(self, focus, frames, pre_frames, post_frames):
        r"""
        metric to reward changing to high deviation frame
        """
        n_frames = focus.shape[0] - 1
        assert frames.shape[0] == n_frames + 1
        assert pre_frames.shape[0] == n_frames
        assert post_frames.shape[0] == n_frames

        focus_diff = np.sum((focus[1:] - focus[:-1])**2, axis=1)
        pre_frame_diff = np.sum(
            (frames[1:] - pre_frames).reshape(n_frames, -1)**2,
            axis=1)
        post_frame_diff = np.sum(
            (frames[:-1] - post_frames).reshape(n_frames, -1)**2,
            axis=1)
        return np.mean(focus_diff * pre_frame_diff * post_frame_diff)


    def __str__(self, prefix=''):
        return (prefix + \
            'SaliencyLoss: frame_dev= %g, focus_dev= %g, ' + \
            'frame_var= %g, belief_dev= %g')%(
            self.frame_dev_coeff, self.focus_dev_coeff,
            self.frame_var_coeff, self.belief_dev_coeff)


# Mutual Information by Change Point Loss: consistancy from known premise
class ActionMICPLoss(FocusChangePointLoss):

    def __init__(self, action_source, *args, **kwargs):
        self.action_source = action_source

        self.mi_match_coeff = kwargs.get('mi_match_coeff', 1.0)
        self.mi_diffs_coeff = kwargs.get('mi_diffs_coeff', 1.0)
        self.verbose = kwargs.get('verbose', False)


    # evaluate changepoint, lesser = more relevant focus
    def forward(self, focus, changepoints, ret_extra=False):
        # get action change points
        premise_cp = self.action_source.get_changepoint()

        # calculate mutual info between two changepoints
        mi_match, mi_diffs, mi_valid = self._mutual_info(focus, changepoints, premise_cp)
        mi_loss = self.mi_diffs_coeff*mi_diffs - self.mi_match_coeff*mi_match
        if self.verbose:
            logger.info('match= %f, diff= %f'%(mi_match, mi_diffs))
        if ret_extra:
            return mi_loss, {'valid': mi_valid}
        return mi_loss


    # correlation between focus change points and premise change points
    def _mutual_info(self, focus, object_cp, premise_cp):
        r"""
        count the number of correlated matching
            x_i = y_j      \rightarrow   match
            x_i \notin y   \rightarrow   diffs
            y_j \notin x   \rightarrow   diffs
        the result is normalized by the number of different change points
        """

        # match up change points with known object's
        n_focus = focus.shape[0]
        match_mask, diffs_mask = util.match_diffs(object_cp, premise_cp, n_focus)

        return np.sum(match_mask)/n_focus, np.sum(diffs_mask)/n_focus, n_focus


    def __str__(self, prefix=''):
        return prefix + 'ActionMICPLoss: mi_match= %g, mi_diffs= %g'%(
            self.mi_match_coeff, self.mi_diffs_coeff)


# Premise-Focus Mutual Information by Change Point Loss: MICP
class PremiseMICPLoss(FocusChangePointLoss):

    def __init__(self, frame_source, premise, *args, **kwargs):
        self.frame_source = frame_source
        self.premise = premise
        self.prox_dist = kwargs.get('prox_dist', np.sqrt(0.5))
        self.batch_size = kwargs.get('batch_size', 100)

        self.mi_match_coeff = kwargs.get('mi_match_coeff', 1.0)
        self.mi_diffs_coeff = kwargs.get('mi_diffs_coeff', 1.0)
        self.mi_valid_coeff = kwargs.get('mi_valid_coeff', 1.0)
        self.verbose = kwargs.get('verbose', False)


    # evaluate changepoint, lesser = more relevant focus
    def forward(self, focus, changepoints, ret_extra=False):
        premise_focus = self._premise_forward()

        # calculate mutual info between two changepoints
        mi_match, mi_diffs, mi_valid = self._mutual_info(focus, premise_focus, changepoints)
        mi_loss = self.mi_diffs_coeff*mi_diffs \
                  - self.mi_match_coeff*mi_match \
                  + self.mi_valid_coeff*mi_valid
        if self.verbose:
            logger.info('match= %f, diff= %f, valid= %f'%(
                        mi_match, mi_diffs, mi_valid))
        if ret_extra:
            return mi_loss, {'valid': mi_valid}
        return mi_loss


    # correlation between focus change points and premise change points
    def _mutual_info(self, object_focus, premise_focus, object_cp):
        r"""
        count the number of correlated matching
            x_i \in prox      \rightarrow   match
            x_i \notin prox   \rightarrow   diffs
        the result is normalized by the number of change points
        """

        # match up change points with known object's
        assert object_focus.shape[0] == premise_focus.shape[0]
        n_focus = object_focus.shape[0]
        object_cp_mask = util.list_to_mask(object_cp, n_focus)

        # filter out those timestamp when object's far from premise
        prox_mask = self._get_prox_mask(object_focus, premise_focus)
        match_mask = object_cp_mask & prox_mask
        diffs_mask = object_cp_mask ^ prox_mask
        if self.verbose:
            logger.info('prox filter valid= %d: match= %3d, diff= %3d'%(
                np.sum(prox_mask), np.sum(match_mask), np.sum(diffs_mask)))

        # TODO: better normalization?
        return np.sum(match_mask)/n_focus, \
               np.sum(diffs_mask)/n_focus, \
               np.sum(prox_mask)/n_focus


    # get known object's focus
    def _premise_forward(self):
        premise_focus = self.premise.forward_all(self.frame_source,
                                                 batch_size=self.batch_size)
        return premise_focus


    # get proximity mask, True => object is close to premise
    def _get_prox_mask(self, object_focus, premise_focus):
    	# TODO: add func to Focus for this
        distances = np.linalg.norm(object_focus-premise_focus, axis=1)
        prox_mask = np.zeros(object_focus.shape[0], dtype=bool)
        prox_mask[distances < self.prox_dist] = True
        return prox_mask


    def __str__(self, prefix=''):
        return prefix + 'PremiseMICPLoss: ' \
            'mi_match= %g, mi_diffs= %g, mi_valid= %g, prox= %g'%(
            self.mi_match_coeff, self.mi_diffs_coeff, self.mi_valid_coeff,
            self.prox_dist)


# MICP with multiple premises
class CollectionMICPLoss(FocusLoss):

    def __init__(self, *micp_losses, **kwargs):
        self.micp_losses = list(micp_losses)
        self.agg_fn = kwargs.get('agg_fn', sum)
        self.cp_detector = kwargs.get('cp_detector', LinearCPD(np.pi/4))


    # evaluate focus loss w.r.t premises
    def forward(self, focus):
        _, changepoints = self.cp_detector.generate_changepoints(focus)
        return self.agg_fn([micp_loss.forward(focus, changepoints)
                           for micp_loss in self.micp_losses])


    # append new premise to the collection
    def append_premise(self, premise):
        self.micp_losses += (premise, )


    # append new premise to the collection
    def replace_premise(self, premise, idx=0):
        self.micp_losses[0] = premise


    # pretty print
    def __str__(self, prefix=''):
        return prefix + 'CollectionMICPLoss: agg_fn= %s\n%s'%(
            self.agg_fn.__str__(),
            '\n'.join(micp_loss.__str__(prefix=prefix+'\t')
                      for micp_loss in self.micp_losses))


# Aggregation of Loss
class CombinedLoss(FocusLoss):

    def __init__(self, *losses, **kwargs):
        self.losses = losses
        self.agg_fn = kwargs.get('agg_fn', sum)


    # evaluate focus loss
    def forward(self, focus):
        return self.agg_fn([loss.forward(focus) for loss in self.losses])


    # pretty print
    def __str__(self, prefix=''):
        return prefix + 'CombinedLoss: agg_fn= %s\n%s'%(
            self.agg_fn.__str__(),
            '\n'.join(loss.__str__(prefix=prefix+'\t') for loss in self.losses))