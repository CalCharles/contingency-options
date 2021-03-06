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
        raise NotImplementedError('forward not implemented')

    def __call__(self, focus):
        return self.forward(focus)


# abstracted class to measure changepoint relevance
class FocusChangePointLoss:

    def forward(self, focus, changepoints, ret_extra=False):
        raise NotImplementedError('forward not implemented')

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
        focus = focus['__train__']
        frames = util.extract_neighbor(
			self.frame_source,
			focus,
            np.arange(focus.shape[0]),
            nb_size=self.nb_size)
        frame_dev = self._deviation_1(frames)
        focus_dev = self._deviation_2(focus)
        stationary_penalty = .01/(.01 + self._deviation_1(focus))
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
            + self.belief_dev_coeff*belief_dev \
            + stationary_penalty


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

        focus_diff = np.sum((focus[1:] - focus[:-1])**2, axis=1)**0.5
        pre_frame_diff = np.sum(
            (frames[1:] - pre_frames).reshape(n_frames, -1)**2,
            axis=1)
        post_frame_diff = np.sum(
            (frames[:-1] - post_frames).reshape(n_frames, -1)**2,
            axis=1)

        # TODO: remove
        # import seaborn as sns; import matplotlib.pyplot as plt
        # try:
        #     fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10)); axes = axes.flatten()
        #     sns.distplot(focus_diff * pre_frame_diff * post_frame_diff, rug=True, ax=axes[0]); axes[0].set_title('focus * pre * post')
        #     sns.distplot(focus_diff * (pre_frame_diff + post_frame_diff), rug=True, ax=axes[1]); axes[1].set_title('focus * (pre + post)')
        #     sns.distplot(focus_diff + pre_frame_diff + post_frame_diff, rug=True, ax=axes[2]); axes[2].set_title('focus + pre + post')
        #     sns.distplot(pre_frame_diff * post_frame_diff, rug=True, ax=axes[3]); axes[3].set_title('pre * post')
        #     plt.show()
        # except:
        #     plt.clf()

        return np.std(focus_diff * (pre_frame_diff + post_frame_diff))


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
        focus = focus['__train__']

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
        self.mi_cndcp_coeff = kwargs.get('mi_cndcp_coeff', 1.0)
        self.verbose = kwargs.get('verbose', False)


    # evaluate changepoint, lesser = more relevant focus
    def forward(self, focus, changepoints, ret_extra=False):
        premise_focus = focus[self.premise]
        train_focus = focus['__train__']

        # calculate mutual info between two changepoints
        loss_vals = self._mutual_info(train_focus, premise_focus, changepoints)
        mi_match, mi_diffs, mi_valid, mi_cndcp = loss_vals
        mi_loss = self.mi_diffs_coeff*mi_diffs \
                  - self.mi_match_coeff*mi_match \
                  + self.mi_valid_coeff*mi_valid \
                  - self.mi_cndcp_coeff*mi_cndcp
        if self.verbose:
            logger.info('match= %f, diffs= %f, valid= %f, cndcp= %f'%(
                        mi_match, mi_diffs, mi_valid, mi_cndcp))
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

        # probability of proximal conditioned on changepoint
        K = 8
        frame_shape_old = self.frame_source.get_shape()[-2:]
        frame_shape = (frame_shape_old[0] // K, frame_shape_old[1] // K)
        is_object_cp = set(object_cp)
        # tp, fp, fn, tn = 1e-10, 1e-10, 1e-10, 1e-10
        cp_cnt = np.zeros(frame_shape)
        prox_cnt = np.zeros(frame_shape)
        cp_prox_cnt = np.zeros(frame_shape)
        total_cnt = np.zeros(frame_shape)
        last_obj_x = None
        for i, obj_fx in enumerate(object_focus):
            obj_x = (obj_fx * frame_shape).astype(int)
            total_cnt[obj_x[0], obj_x[1]] += 1
            if prox_mask[i]:
                # if i == 0 or not prox_mask[i-1] or any(last_obj_x != obj_x):
                #     last_obj_x = obj_x
                prox_cnt[obj_x[0], obj_x[1]] += 1
            #     if i in is_object_cp:
            #         tp += 1
            #     else:
            #         fp += 1
            # else:
            #     if i in is_object_cp:
            #         fn += 1
            #     else:
            #         tn += 1
        for cpi in object_cp:
            obj_x = (object_focus[cpi] * frame_shape).astype(int)
            premise_x = (premise_focus[cpi] * frame_shape).astype(int)
            cp_cnt[obj_x[0], obj_x[1]] += 1
            if prox_mask[cpi]:
                cp_prox_cnt[obj_x[0], obj_x[1]] += 1
        none_cnt = total_cnt - cp_cnt - prox_cnt + cp_prox_cnt
        # cnt_valid = np.logical_and(cp_prox_cnt > 0.0, 
        #                            prox_cnt - cp_prox_cnt > 0.0)
        cnt_valid = cp_cnt + prox_cnt > 0.0
        cndcp_score = np.mean(2 * cp_prox_cnt[cnt_valid]
                              / (cp_cnt[cnt_valid] + prox_cnt[cnt_valid])) \
                              if np.sum(cnt_valid) > 7.0 else 0.0
        logger.info('cnt_valid= %g'%(np.sum(cnt_valid)))
        # cnt_valid = prox_cnt > 0.0
        # cndcp_score = np.mean(cp_prox_cnt[cnt_valid]
        #                       / prox_cnt[cnt_valid]) \
        #                       if np.sum(cnt_valid) > 0.0 else 0.0
        # util.confuse_metrics(cp_prox_cnt, prox_cnt-cp_prox_cnt, 
        #                      cp_cnt-cp_prox_cnt, none_cnt)
        
        # print('tp', int(tp), 'fp', int(fp), 'fn', int(fn), 'tn', int(tn))
        # util.confuse_metrics(tp, fp, fn, tn)

        # import cv2
        # util.count_imsave('tp.png', cv2.resize(cp_prox_cnt, dsize=frame_shape_old, interpolation=cv2.INTER_NEAREST), cm=np.array([0.3, 1.0, 0.3]))
        # util.count_imsave('fp.png', cv2.resize(prox_cnt - cp_prox_cnt, dsize=frame_shape_old, interpolation=cv2.INTER_NEAREST), cm=np.array([1.0, 0.3, 0.3]))
        # util.count_imsave('fn.png', cv2.resize(cp_cnt - cp_prox_cnt, dsize=frame_shape_old, interpolation=cv2.INTER_NEAREST), cm=np.array([0.3, 0.3, 1.0]))

        return np.sum(match_mask)/n_focus, \
               np.sum(diffs_mask)/n_focus, \
               np.sum(prox_mask)/n_focus, \
               cndcp_score


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
            'mi_match= %g, mi_diffs= %g, mi_valid= %g, mi_cndcp= %g, prox= %g'%(
            self.mi_match_coeff, self.mi_diffs_coeff, self.mi_valid_coeff,
            self.mi_cndcp_coeff, self.prox_dist)


# MICP with multiple premises
class CollectionMICPLoss(FocusLoss):

    def __init__(self, micp_losses, **kwargs):
        self.micp_losses = list(micp_losses)
        self.agg_fn = kwargs.get('agg_fn', sum)
        self.cp_detector = kwargs.get('cp_detector', LinearCPD(np.pi/4))


    # evaluate focus loss w.r.t premises
    def forward(self, focus):
        focus_x = focus['__train__']
        _, changepoints = self.cp_detector.generate_changepoints(focus_x)
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


# attention mapper
def attn_map(attn_1, attn_2):
    raise NotImplementedError('not finished')


# changepoint loss for attention models
class AttentionPremiseMICPLoss:

    def __init__(self, frame_source, premise, *args, **kwargs):
        self.frame_source = frame_source
        self.premise = premise  # produce focus
        self.prox_dist = kwargs.get('prox_dist', 0.1)
        self.attn_t = kwargs.get('attn_t', 0.5)  # threshold for being object

        self.mi_match_coeff = kwargs.get('mi_match_coeff', 1.0)
        self.mi_diffs_coeff = kwargs.get('mi_diffs_coeff', 1.0)
        self.active_attn_coeff = kwargs.get('active_attn_coeff', 1.0)
        self.verbose = kwargs.get('verbose', False)


    # evaluate focus loss
    def forward(self, focus, attn):
        train_attn = attn['__train__']
        # train_attn = 1.0 / (1 + np.exp(-train_attn))

        mi_match, mi_diffs = 0.0, 0.0
        if self.mi_match_coeff > 0 or self.mi_diffs_coeff > 0:
            premise_focus = focus[self.premise]
            mi_match, mi_diffs = self._mutual_info(train_attn, premise_focus)
        active_attn = np.mean(train_attn)
        mi_loss = self.mi_diffs_coeff*mi_diffs \
                  - self.mi_match_coeff*mi_match \
                  + self.active_attn_coeff*active_attn
        if self.verbose:
            logger.info('match= %f, diffs= %f, active= %f'%(
                        mi_match, mi_diffs, active_attn))
        return mi_loss


    # evaluate focus loss
    def _mutual_info(self, train_attn, premise_focus):
        n_focus = train_attn.shape[0] - 1

        # get changepoint in attention intensity
        object_mask = train_attn > self.attn_t
        object_cp_mask = object_mask[1:] != object_mask[:-1]

        # get proximity mask
        prox_mask = self._get_prox_mask(object_mask, premise_focus)[:-1]

        # compute match and diffs
        prox_cnt = prox_mask.sum(axis=1).sum(axis=1)
        v_idx = prox_cnt > 0
        match_cnt = (object_cp_mask & prox_mask).sum(axis=1).sum(axis=1)
        diffs_cnt = (object_cp_mask ^ prox_mask).sum(axis=1).sum(axis=1)

        # import matplotlib.pyplot as plt
        # i = 0
        # fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(15, 3))
        # for m, d in zip(object_cp_mask, prox_mask):
        #     axes[0, i].imshow(m[0])
        #     axes[0, i].axis('off')
        #     axes[1, i].imshow(d[0])
        #     axes[1, i].axis('off')
        #     i = (i+1) % 10
        #     if i == 0:
        #         plt.show()
        #         fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(15, 3))

        return np.sum(match_cnt[v_idx] / prox_cnt[v_idx]) / n_focus, \
               np.sum(diffs_cnt[v_idx] / prox_cnt[v_idx]) / n_focus


    # get object proximity mask
    def _get_prox_mask(self, object_mask, premise_focus):
        premise_dist = util.focus2attn(
            premise_focus, 
            self.frame_source.get_shape()[-2:],
            fn=util.conic)
        return object_mask & (premise_dist < self.prox_dist)


    # pretty print
    def __str__(self, prefix=''):
        return (prefix + 'AttentionPremiseMICPLoss: ' + \
            'prox_dist= %g, attn_t= %g, ' + \
            'mi_match= %g, mi_diffs= %g, ' + \
            'active_attn= %g')%(
            self.prox_dist, self.attn_t,
            self.mi_match_coeff, self.mi_diffs_coeff,
            self.active_attn_coeff)


# Aggregation of Loss
class CombinedLoss(FocusLoss):

    def __init__(self, focus_losses, attn_losses=[], **kwargs):
        self.focus_losses = focus_losses
        self.attn_losses = attn_losses
        self.agg_fn = kwargs.get('agg_fn', sum)


    # evaluate focus loss
    def forward(self, focus):
        if isinstance(focus, tuple) and len(focus) == 2:
            focus, attn = focus
            return self.agg_fn(
                [loss.forward(focus) for loss in self.focus_losses] +
                [loss.forward(focus, attn) for loss in self.attn_losses])

        # else...
        return self.agg_fn([loss.forward(focus) for loss in self.focus_losses])


    # pretty print
    def __str__(self, prefix=''):
        fstrs = [loss.__str__(prefix=prefix+'\t') for loss in self.focus_losses]
        astrs = [loss.__str__(prefix=prefix+'<a>\t')
                 for loss in self.attn_losses]
        return prefix + 'CombinedLoss: agg_fn= %s\n%s\n%s'%(
            self.agg_fn.__str__(), '\n'.join(fstrs), '\n'.join(astrs))