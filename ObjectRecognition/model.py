import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial
from Models.models import pytorch_model

import logging
logging.basicConfig(format='%(levelname)s [%(asctime)s]: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

from ObjectRecognition.dataset import Dataset
import ObjectRecognition.util as util


ACTIVATIONS = {
    'Threshold': nn.Threshold,
    'ReLU': nn.ReLU,
    'RReLU': nn.RReLU,
    'Hardtanh': nn.Hardtanh,
    'ReLU6': nn.ReLU6,
    'Sigmoid': nn.Sigmoid,
    'Tanh': nn.Tanh,
    'ELU': nn.ELU,
    'SELU': nn.SELU,
    'GLU': nn.GLU,
    'Hardshrink': nn.Hardshrink,
    'LeakyReLU': nn.LeakyReLU,
    'LogSigmoid': nn.LogSigmoid,
    'Softplus': nn.Softplus,
    'Softshrink': nn.Softshrink,
    'PReLU': nn.PReLU,
    'Softsign': nn.Softsign,
    'Tanhshrink': nn.Tanhshrink,
    'Softmin': nn.Softmin,
    'Softmax': nn.Softmax,
    'Softmax2d': nn.Softmax2d,
    'LogSoftmax': nn.LogSoftmax,
}



# load parameter from file
def get_param_path(directory, name, ext=None):
    if ext is None:
        # try numpy (npy) and torch (pth)
        np_path = os.path.join(directory, '%s.npy'%name)
        t_path = os.path.join(directory, '%s.pth'%name)
        if os.path.isfile(np_path):
            ext = 'npy'
        elif os.path.isfile(t_path):
            ext = 'pth'
        else:
            raise FileNotFoundError('no param file %s found in directory %s'%
                                    (name, directory))

    # param wrt file extension
    return os.path.join(directory, '%s.%s'%(name, ext))


# load parameter from file
def load_param(param_path, ext=None):
    if isinstance(param_path, tuple):
        param_path = get_param_path(param_path[0], param_path[1], ext)
    if ext is None: ext = os.path.splitext(param_path)[1][1:]
    if ext == 'npy':
        return np.load(param_path)
    elif ext == 'pth':
        return torch.load(param_path)
    raise ValueError('file extension %s not supported'%(ext))


class ModelObject(nn.Module):

    def __init__(self, *args, **kwargs):
        super(ModelObject, self).__init__()
        self.use_prior = False
        self.parameter_count = -1

        # preprocessing
        self.binarize = kwargs.get('binarize', None)

    # set parameter corresponding to given param type
    def set_parameters(self, param):
        if isinstance(param, np.ndarray):
            # numpy array
            self.set_parameters_np(param)
        elif isinstance(param, dict):
            # torch state dict
            self.set_parameters_torch(param)


    # set parameters in order of name-size pair, param_np is a indicable list
    def set_parameters_np(self, param_np):
        if len(param_np) != self.count_parameters():
            raise ValueError('invalid number of parameters to set')
        pval_idx = 0
        for param in self.parameters():
            param_size = np.prod(param.size())
            cur_param_np = param_np[pval_idx : pval_idx+param_size]
            cuda = False
            if param.is_cuda:
                cuda = True
            param.data = torch.from_numpy(cur_param_np) \
                              .reshape(param.size()).float()
            if cuda:
                param.data = param.data.cuda()
            pval_idx += param_size


    # set parameters from torch state dict
    def set_parameters_torch(self, param_dict):
        self.load_state_dict(param_dict)


    # count number of parameters
    def count_parameters(self, reuse=True):
        if reuse and self.parameter_count > 0:
            return self.parameter_count
        self.parameter_count = 0
        for param in self.parameters():
            self.parameter_count += np.prod(param.size())
        return self.parameter_count


    # Run forward on model with full dataset from game environment
    def forward_all(self, game_env, batch_size=100, ret_extra=False):
        outputs = np.zeros((game_env.n_state,)+self.output_size(), dtype=float)
        extra = np.array([])
        if self.use_prior:
            prev_out = None
            for i in range(game_env.n_state):
                frames = game_env.get_frame(i, i+1)  # batch format
                frames = self.preprocess(frames)
                forward_out = self.forward(frames,
                                        prev_out=prev_out, 
                                        ret_numpy=True,
                                        ret_extra=ret_extra)
                if ret_extra:
                    extra = np.vstack([extra, forward_out[1]]) \
                            if extra.size else forward_out[1]
                    forward_out = forward_out[0]
                prev_out = forward_out
                outputs[i, ...] = forward_out
        else:
            for l in range(0, game_env.n_state, batch_size):
                r = min(l + batch_size, game_env.n_state)
                frames = game_env.get_frame(l, r)
                frames = self.preprocess(frames)
                forward_out = self.forward(frames,
                                        ret_numpy=True,
                                        ret_extra=ret_extra)
                if ret_extra:
                    extra = np.vstack([extra, forward_out[1]]) \
                            if extra.size else forward_out[1]
                    forward_out = forward_out[0]
                outputs[l:r, ...] = forward_out

        if ret_extra:
            return outputs, extra
        return outputs


    # preprocess input
    def preprocess(self, frames):
        if self.binarize:
            frames = util.binarize(frames, self.binarize)
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames).float()
        if next(self.parameters()).is_cuda:
            frames = frames.cuda()
        return frames


"""
Focus models: given a image, return a coordinate of focus center
    image_shape:    (width, height)
    image:          array of size (width, height), each element in [0, 1]
    focus:          array of two integers, in domain [0, width) x [0, height)
"""


class ModelFocusInterface(nn.Module):

    # initialize network propeties
    def init_net(self):
        raise NotImplementedError


    # push input forward
    @torch.no_grad()
    def forward(self, img, prev_out=None):
        raise NotImplementedError


    # input size of this network
    def input_size(self):
        raise NotImplementedError


    # output size of this network
    def output_size(self):
        raise NotImplementedError


    # count number of parameters
    def count_parameters(self, reuse=True):
        raise NotImplementedError


    # Run forward on model with full dataset from game environment
    def forward_all(self, dataset, batch_size=100):
        raise NotImplementedError


    # set parameters in order of name-size pair, param_val is a indicable list
    def set_parameters(self, param_val):
        raise NotImplementedError


# gaussian pdf centered at (0, 0) with signma
def gaussian(x, y, sigma):
    return np.exp(-(x**2 + y**2)/(2 * sigma**2)) / (2 * np.pi * sigma)


# generate focus prior filter
# prevs has shape of (batch, 2) each entry the previous position
def prior_filter(prevs, shape):
    assert shape[0] == prevs.shape[0]
    filters = np.zeros(shape)
    for i, prev in enumerate(prevs):
        xs = np.linspace(0, 1, shape[2]) - prev[0]
        ys = np.linspace(0, 1, shape[3]) - prev[1]
        if 0 <= prev[0] <= 1 or 0 <= prev[1] <= 1:
            filters[i, ...] = gaussian(xs[:, None], ys[None, :], 1.0)
        else:  # invalid prior, might be the beginning?
            filters[i, ...] = 1.0
    return filters


class ModelFocusCNN(ModelObject, ModelFocusInterface):
    def __init__(self, image_shape, net_params, *args, **kwargs):
        super(ModelFocusCNN, self).__init__(*args, **kwargs)

        # interface parameters
        if len(image_shape) == 2:
            self.input_shape = image_shape
            self.input_channel = 1
        elif len(image_shape) == 3:
            self.input_shape = (image_shape[0], image_shape[1])
            self.input_channel = image_shape[2]
        else:
            raise ValueError("only support duplet or triplet image_shape")
        self.input_shape_flat = np.prod(self.input_shape)

        # network construction parameters
        self.net_params = net_params
        self.use_prior = kwargs.get('use_prior', False)
        self.argmax_mode = kwargs.get('argmax_mode', 'first')  # 'first', 'rand'

        self.init_net()


    # initialize network propeties
    def init_net(self):
        # convolutional layers
        cur_channel = self.input_channel
        sublayers = []
        for i in range(self.net_params['filter']):
            c = self.net_params['channel'][i]
            k = self.net_params['kernel_size'][i]
            s = self.net_params['stride'][i]
            p = self.net_params['padding'][i]
            a = self.net_params['activation_fn'][i]
            sublayers.append(nn.Conv2d(cur_channel, c, kernel_size=k, stride=s, padding=p))
            sublayers.append(nn.BatchNorm2d(c))
            sublayers.append(ACTIVATIONS[a]())
            sublayers.append(nn.MaxPool2d(kernel_size=k, stride=s))
            cur_channel = c
        self.layers = nn.Sequential(*sublayers)
        self.parameter_count = -1


    # push input forward
    @torch.no_grad()
    def forward(self, img, prev_out=None, ret_numpy=True, ret_extra=False):
        out = img
        for layer in self.layers:
            out = layer(out).detach()
        if prev_out is not None:  # apply prior filter if specified
            pfilter = prior_filter(prev_out, out.size())
            pfilter = self.preprocess(pfilter)
            out = torch.mul(out, pfilter)
        focus_out = self.argmax_xy(out)

        focus_out = focus_out if ret_numpy \
                    else torch.from_numpy(focus_out).float()
        if ret_extra:
            return focus_out, pytorch_model.unwrap(out)
        return focus_out


    # pick max coordinate
    def argmax_xy(self, out):
        out = pytorch_model.unwrap(out)
        batch_size = out.shape[0]
        row_size = out.shape[2]
        col_size = out.shape[3]
        
        if self.argmax_mode == 'first':
            # first argmax
            argmax = np.argmax(out.reshape((batch_size, -1)), axis=1)
        elif self.argmax_mode == 'rand':
            # random argmax for tie-breaking
            out = out.reshape((batch_size, -1))
            out_max = np.max(out, axis=1)
            argmax = np.array([np.random.choice(np.flatnonzero(line == line_max)) 
                               for line, line_max in zip(out, out_max)])
        else:
            raise ValueError('argmax_mode %s invalid'%(self.argmax_mode))
        
        argmax %= row_size * col_size  # in case of multiple filters
        argmax_coor = np.array([np.unravel_index(argmax_i, (row_size, col_size)) 
                                for argmax_i in argmax], dtype=float)
        argmax_coor = argmax_coor / np.array([row_size, col_size])
        return argmax_coor


    # input size of this network
    def input_size(self):
        return (self.input_channel,) + self.input_shape


    # output size of this network
    def output_size(self):
        return (2, )


    # pretty print
    def __str__(self, prefix=''):
        return prefix + 'FocusCNN: ' \
            'input shape= %s, channel= %d, flat= %d, net_params= %s'%(
            str(self.input_shape), self.input_channel, self.input_shape_flat,
            self.net_params)


# focus model using mean and variance
class ModelFocusMeanVar(ModelObject, ModelFocusInterface):
    def __init__(self, img_mean, img_var, *args, **kwargs):
        super(ModelFocusMeanVar, self).__init__(*args, **kwargs)

        # mean and variance
        assert img_mean.shape == img_var.shape
        assert len(img_mean.shape) == 3
        self.img_mean = img_mean
        self.img_var = img_var
        self.match_shape = self.img_mean.shape
        self.pad_shape = ((self.match_shape[-2]+1)//2, (self.match_shape[-1]+1)//2)

        # extra...
        self.argmax_mode = kwargs.get('argmax_mode', 'first')  # 'first', 'rand'


    # push input forward
    @torch.no_grad()
    def forward(self, imgs, prev_out=None, ret_numpy=True, ret_extra=False):
        match_field = np.zeros((imgs.shape[0],) + imgs.shape[-2:])
        for i in range(imgs.shape[0]):  # batch
            for j in range(imgs.shape[1]):  # channel
                pad_frame = np.pad(imgs[i, j], self.pad_shape, 'constant')
                for x in range(imgs.shape[2]):
                    for y in range(imgs.shape[3]):
                        match_field[i, x, y] += self.match(
                            pad_frame[None, x:x+self.match_shape[-2],
                                      y:y+self.match_shape[-1]]
                        )
                # match_field[i] = np.exp(match_field[i] - np.max(match_field[i]))
                # match_field[i] = match_field[i] / np.sum(match_field[i])

        focus_out = self.argmax_xy(match_field)
        focus_out = focus_out if ret_numpy \
                    else torch.from_numpy(focus_out).float()
        if ret_extra:
            return focus_out, match_field
        return focus_out


    # matching function
    def match(self, img):
        # self.img_mean[:, :10, :] = None
        # self.img_mean[:, :, :10] = None
        sel_bool = ~np.isnan(self.img_mean)
        img_sel = img[sel_bool]
        mean_sel = self.img_mean[sel_bool]
        var_sel = self.img_var[sel_bool]
        return -np.sum((img_sel - mean_sel)**2 / var_sel)
        # return -np.sum((img_sel - mean_sel)**2)


    # pick max coordinate
    def argmax_xy(self, out):
        # return util.argmax2d(out, self.argmax_mode)
        batch_size = out.shape[0]
        row_size = out.shape[1]
        col_size = out.shape[2]
        
        if self.argmax_mode == 'first':
            # first argmax
            argmax = np.argmax(out.reshape((batch_size, -1)), axis=1)
        elif self.argmax_mode == 'rand':
            # random argmax for tie-breaking
            out = out.reshape((batch_size, -1))
            out_max = np.max(out, axis=1)
            argmax = np.array([np.random.choice(np.flatnonzero(line == line_max)) 
                               for line, line_max in zip(out, out_max)])
        else:
            raise ValueError('argmax_mode %s invalid'%(self.argmax_mode))
        
        argmax %= row_size * col_size  # in case of multiple filters
        argmax_coor = np.array([np.unravel_index(argmax_i, (row_size, col_size)) 
                                for argmax_i in argmax], dtype=float)
        argmax_coor = argmax_coor / np.array([row_size, col_size])
        return argmax_coor


    # train frmo focus model
    @staticmethod
    def from_focus_model(focus_model_forward, dataset, nb_size, batch_size=100):
        n_channel = 1  # dataset.get_shape()[-3]  # TODO: properly do this
        match_size = (n_channel, nb_size[0]*2+1, nb_size[1]*2+1)
        img_stack = np.zeros((dataset.n_state,) + match_size)
        weight_stack = np.zeros(dataset.n_state)
        cur_n = 0

        for l in range(0, dataset.n_state, batch_size):
            print(l)  # TODO: remove
            r = min(l + batch_size, dataset.n_state)
            frames = dataset.get_frame(l, r)
            focus, extra = focus_model_forward(frames)
            neighbor = util.extract_neighbor(dataset, focus, np.arange(l, r),
                                             nb_size=nb_size)
            img_stack[l:r, 0] = neighbor  # properly channels?
            weight_stack[l:r] = util.focus_intensity(focus[l:r], extra[l:r])

        img_mean = np.average(img_stack, weights=weight_stack, axis=0)
        img_var = np.average((img_stack - img_mean)**2, weights=weight_stack, 
                             axis=0)**0.5  # TODO: this is biased
        return ModelFocusMeanVar(img_mean, img_var)


    # pretty print
    def __str__(self, prefix=''):
        return prefix + 'FocusMeanVar:'


# Boosting from multiple neuron network based on change points
class ModelFocusBoost(ModelObject, ModelFocusInterface):
    def __init__(self, cp_detector, *models, **kwargs):
        super(ModelFocusBoost, self).__init__(*args, **kwargs)

        # save models in sequential hierarchy
        self.models = models
        self.cp_detector = cp_detector
        self.train_flags = kwargs.get('train_flags', [False]*len(self.models))

        self.sanitize()


    # push input forward
    @torch.no_grad()
    def forward(self, imgs, batch_size=100):
        n_img = imgs.shape[0]
        outputs = np.zeros((n_img,) + self.output_size(), dtype=float)

        # loop for each model sequentially
        cp_idxs = np.arange(n_img)  # pick all point
        for i, model in enumerate(self.models):

            # forward for focus at this model
            for l in range(0, len(cp_idxs), batch_size):
                r = min(l+batch_size, len(cp_idxs))
                t_idxs = cp_idxs[l:r]
                outputs[t_idxs] = model.forward(imgs[t_idxs])

            # compute change points and fill in boolean mask
            if i < len(self.models) - 1:
                changepoints = self.cp_detector.generate_changepoints(outputs)
                cp_mask = np.zeros(n_img, dtype=bool)
                cp_mask[changepoints] = True
                cp_idxs = np.where(cp_mask)
            
        return outputs


    # input size of this network
    def input_size(self):
        return self.models[0].input_size()


    # output size of this network
    def output_size(self):
        return (2, )


    # get list of parameters like nn.Module's
    def parameters(self):
        return sum(map(lambda m: list(m.parameters()),
                       self.get_trainable()), [])


    # count number of parameters
    def count_parameters(self):
        return sum(map(lambda m: m.count_parameters(), self.get_trainable()))


    # Run forward on model with full dataset from game environment
    def forward_all(self, game_env, batch_size=100):
        # TODO: proper game_env get frames interface
        all_frames = game_env.get_frame(0, game_env.n_state)
        out = self.forward(torch.from_numpy(all_frames).float(),
                           batch_size=100)
        return out


    # get list of trainable models
    def get_trainable(self):
        return [model for midx, model in enumerate(self.models) 
                if self.train_flags[midx]]


    # check if connections make sense
    def sanitize(self):
        assert len(self.models) > 0
        assert all(model.input_size() == self.models[0].input_size()
                   for model in self.models)
        assert all(model.output_size() == (2, )
                   for model in self.models)


    # pretty print
    def __str__(self, prefix=''):
        return prefix + 'FocusCPBoost:\n%s'%('\n'.join(
            model.__str__(prefix=prefix+'\t')
            for model in self.models))


"""
Attention models: given a image, return a same-size attention intensity
    image_shape:    (width, height)
    image:          array of size (width, height), each element in [0, 1]
    focus:          array of two integers, in domain [0, width) x [0, height)
"""
# TODO: refactor focus into attention

class ModelAttentionCNN(ModelObject):
    def __init__(self, image_shape, net_params, *args, **kwargs):
        super(ModelAttentionCNN, self).__init__(*args, **kwargs)

        # interface parameters
        if len(image_shape) == 2:
            self.input_shape = image_shape
            self.input_channel = 1
        elif len(image_shape) == 3:
            self.input_shape = (image_shape[0], image_shape[1])
            self.input_channel = image_shape[2]
        else:
            raise ValueError("only support duplet or triplet image_shape")
        self.input_shape_flat = np.prod(self.input_shape)

        # network construction parameters
        self.net_params = net_params
        self.init_net()

    # initialize network propeties
    def init_net(self):
        # convolutional layers
        cur_channel = self.input_channel
        sublayers = []
        for i in range(self.net_params['filter']):
            c = self.net_params['channel'][i]
            k = self.net_params['kernel_size'][i]
            s = self.net_params['stride'][i]
            p = self.net_params['padding'][i]
            a = self.net_params['activation_fn'][i]
            sublayers.append(nn.Conv2d(cur_channel, c, kernel_size=k, stride=s, padding=p))
            sublayers.append(nn.BatchNorm2d(c))
            sublayers.append(ACTIVATIONS[a]())
            sublayers.append(nn.MaxPool2d(kernel_size=k, stride=s))
            cur_channel = c
        self.layers = nn.Sequential(*sublayers)
        self.parameter_count = -1


    # push input forward
    def forward(self, img, ret_numpy=False, ret_extra=False):
        # out = img
        out = self.preprocess(img)
        for layer in self.layers:
            out = layer(out)
        return out if not ret_numpy else pytorch_model.unwrap(out)


    # input size of this network
    def input_size(self):
        return (self.input_channel,) + self.input_shape


    # output size of this network
    def output_size(self):
        return self.input_size()


    # train with focus model (smoothening)
    def from_focus_model(self, focus_model_forward, dataset, *args, **kwargs):
        lr = kwargs.get('lr', 1e-3)
        n_iter = kwargs.get('n_iter', 100)
        epoch = kwargs.get('epoch', None)
        if epoch is not None:
            logger.info('EPOCH %3d: lr= %f, n_iter= %d'%(epoch, lr, n_iter))

        # get target attention
        if isinstance(dataset, Dataset):
            print('WARNING: processing whole dataset in one batch!')
            frames = dataset.get_frame(0, dataset.n_state)
        else:
            frames = dataset
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames).float()
        focus, intensity = focus_model_forward(frames, ret_extra=True)
        focus_attn = util.focus2attn(
            focus,
            self.input_shape,
            fn=partial(util.gaussian_pdf, normalized=False))
        # focus_attn = 10 * focus_attn - 1  # rescale
        focus_attn = torch.from_numpy(focus_attn).float()
        focus_weight = util.focus_intensity(focus, intensity)
        focus_weight = torch.from_numpy(focus_weight).float()
        focus_weight_sum = focus_weight.sum()
        if next(self.parameters()).is_cuda:
            focus_attn = focus_attn.cuda()
            focus_weight = focus_weight.cuda()
            focus_weight_sum = focus_weight_sum.cuda()
        # train
        lda_1 = 2 # attention regularization
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for t in range(n_iter):
            output = self(frames)
            # loss = (output - focus_attn).pow(2).mean() \
            #        + (lda_1 + 1) * output.abs().mean()
            loss = -(focus_weight 
                     * (util.nn_logsoftmax(output) * focus_attn).mean(dim=(-1, -2, -3))
                   ).sum() / focus_weight_sum
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.info('iteration %2d: loss= %f'%(t, loss.item()))


    # pretty print
    def __str__(self, prefix=''):
        return prefix + 'AttentionCNN: ' \
            'input shape= %s, channel= %d, flat= %d, net_params= %s'%(
            str(self.input_shape), self.input_channel, self.input_shape_flat,
            self.net_params)


"""
DAG Model: handle model evaluation in direct acyclic graph
"""

# reverse directed graph
def reverse_dgraph(dg):
    rdg = {u: [] for u in dg.keys()}
    for u, vs in dg.items():
        for v in vs:
            rdg[v].append(u)
    return rdg


# toposort iterator, dag is dict(u: [v1, v2, ...])
def toposort(dag):
    rg = reverse_dgraph(dag)
    dep_cnt = {u: len(vs) for u, vs in reverse_dgraph(dag).items()}
    free_nodes = [u for u, dc in dep_cnt.items() if dc == 0]
    while len(free_nodes) > 0:
        # get a free node
        cur_u = free_nodes.pop()
        yield cur_u

        # update dependency
        for v in dag[cur_u]:
            dep_cnt[v] -= 1
            if dep_cnt[v] == 0: free_nodes.append(v)


# for model evaluation in DAG fashion
class ModelCollectionDAG():

    def __init__(self, *args, **kwargs):
        self.model_list = dict()  # model collection
        self.model_augment_pt = dict()  # postprocessing after forward
        self.model_augment_fn = dict()  # operation after forward
        self.model_dep = dict()  # dependency graph
        self.model_dir = dict()  # evaluation direction graph
        self.augment_combine = kwargs.get('augment_combine', util.pick_first)
        self.iscuda = False

        # model states
        self.train_model_id = None  # for set_parameters


    # add model node to the graph
    # e.g. add_model('ball', ball_model, ['paddle'])
    def add_model(self, model_id, model, dep, 
                  augment_pt=util.noop_y,
                  augment_fn=util.noop_x):
        if model_id in self.model_list.keys():
            raise ValueError('model %s already exists'%model_id)
        for d_model_id in dep:
            if d_model_id not in self.model_list.keys():
                raise ValueError('dependency model "%s" unknown'%d_model_id)
        self.model_list[model_id] = model
        self.model_augment_pt[model_id] = augment_pt
        self.model_augment_fn[model_id] = augment_fn
        self.model_dep[model_id] = dep
        self.model_dir[model_id] = []
        for d_model_id in dep:
            self.model_dir[d_model_id].append(model_id)

    def cuda(self):
        print(list(self.model_list.keys()))
        for key in self.model_list.keys():
            self.model_list[key] = self.model_list[key].cuda()
            print(key, self.model_list[key], next(self.model_list[key].parameters()).is_cuda)
        self.iscuda = True
        return self

    def cpu(self):
        for key in self.model_list.keys():
            self.model_list[key] = self.model_list[key].cpu()
        self.iscuda = False
        return self

    # set trainable model, only supported one trainable model
    def set_trainable(self, model_id):
        if model_id not in self.model_list.keys():
            raise ValueError('model %s unknown'%model_id)
        self.train_model_id = model_id


    # set parameter of the trainable model
    def set_parameters(self, params):
        if self.train_model_id is None:
            raise ValueError('no trainable model')
        self.model_list[self.train_model_id].set_parameters(params)


    # count parameters of the trainable model
    def count_parameters(self):
        if self.train_model_id is None:
            raise ValueError('no trainable model')
        return self.model_list[self.train_model_id].count_parameters()


    # forward
    @torch.no_grad()
    def forward(self, img, ret_numpy=False, ret_extra=False):
        outs = dict()
        extras = dict()
        augment_img = dict()
        for model_id in toposort(self.model_dir):
            cur_model = self.model_list[model_id]
            cur_augment_pt = self.model_augment_pt[model_id]
            cur_augment_fn = self.model_augment_fn[model_id]

            # get input image
            prev_img = [augment_img[d_model_id] 
                        for d_model_id in self.model_dep[model_id]]
            if len(prev_img) == 0:
                cur_img = img
            else:
                cur_img = self.augment_combine(prev_img)

            # forward the model
            # print(cur_img)
            cur_img = cur_model.preprocess(cur_img)
            # print(model_id, cur_img.is_cuda)
            # if self.iscuda and not cur_img.is_cuda:
            #     cur_img = cur_img.cuda()
            outs[model_id], extras[model_id] = cur_model.forward(cur_img,
                ret_numpy=ret_numpy,
                ret_extra=True)
            # if ret_extra:
            #     outs[model_id], extras[model_id] = cur_model.forward(cur_img,
            #         ret_numpy=ret_numpy,
            #         ret_extra=ret_extra)
            # else:
            #     outs[model_id] = cur_model.forward(cur_img,
            #         ret_numpy=ret_numpy,
            #         ret_extra=ret_extra)
            outs[model_id] = cur_augment_pt(extras[model_id], outs[model_id])

            # update augmented image
            augment_img[model_id] = cur_augment_fn(cur_img, outs[model_id])

        if self.train_model_id:
            outs['__train__'] = outs[self.train_model_id]
            if ret_extra:
                extras['__train__'] = extras[self.train_model_id]
        if ret_extra:
            return outs, extras
        return outs


    # forward on full dataset from game environment
    def forward_all(self, game_env, batch_size=100, ret_extra=False):
        outs = None
        extras = None
        for l in range(0, game_env.n_state, batch_size):
            # run forward on this batch
            r = min(l + batch_size, game_env.n_state)
            frames = game_env.get_frame(l, r)
            frames = torch.from_numpy(frames).float()
            if ret_extra:
                forward_outs = self.forward(frames, ret_numpy=True, 
                                            ret_extra=ret_extra)
                forward_outs, forward_extras = forward_outs
            else:
                forward_outs = self.forward(frames, ret_numpy=True, 
                                            ret_extra=ret_extra)

            # combine outputs
            if outs is None:
                outs = forward_outs
                if ret_extra:
                    extras = forward_extras
            else:
                for model_id in outs.keys():
                    outs[model_id] = np.vstack((outs[model_id], 
                                                forward_outs[model_id]))
                    if ret_extra:
                        extras[model_id] = np.vstack((
                            extras[model_id],
                            forward_extras[model_id]))
        if ret_extra:
            return outs, extras
        return outs


    # pretty print
    def __str__(self, prefix=''):
        model_lines = []
        for model_id in self.model_list:
            m = self.model_list[model_id]
            model_lines.append(m.__str__(prefix=prefix+'\t'))
        return prefix + 'ModelCollectionDAG: ' \
            'dependency: %s\n%s'%(
            str(self.model_dep), '\n'.join(model_lines))