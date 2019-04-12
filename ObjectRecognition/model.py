import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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


class ModelFocus(ModelFocusInterface):

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
            param.data = torch.from_numpy(cur_param_np) \
                              .reshape(param.size()).float()
            pval_idx += param_size


    # set parameters from torch state dict
    def set_parameters_torch(self, param_dict):
        self.load_state_dict(param_dict)


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


class ModelFocusCNN(ModelFocus):
    def __init__(self, image_shape, net_params, *args, **kwargs):
        super(ModelFocusCNN, self).__init__()

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
    def forward(self, img, prev_out=None, ret_extra=False):
        out = img
        for layer in self.layers:
            out = layer(out)
        if prev_out is not None:  # apply prior filter if specified
            pfilter = prior_filter(prev_out, out.size())
            pfilter = torch.from_numpy(pfilter).float()
            out = torch.mul(out, pfilter)
        focus_out = self.argmax_xy(out)

        if ret_extra:
            return focus_out, out.detach().numpy()
        return focus_out


    # pick max coordinate
    def argmax_xy(self, out):
        batch_size = out.size(0)
        row_size = out.size(2)
        col_size = out.size(3)
        
        if self.argmax_mode == 'first':
            # first argmax
            mx, argmax = out.reshape((batch_size, -1)).max(1)
        elif self.argmax_mode == 'rand':
            # random argmax for tie-breaking
            out = out.reshape((batch_size, -1))
            argmax = np.array([np.random.choice(np.flatnonzero(line == line_max)) 
                               for line, line_max in zip(out, out.max(1)[0])])
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
                frames = torch.from_numpy(frames).float()
                forward_out = self.forward(frames, prev_out=prev_out, 
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
                frames = torch.from_numpy(frames).float()
                forward_out = self.forward(frames, ret_extra=ret_extra)
                if ret_extra:
                    extra = np.vstack([extra, forward_out[1]]) \
                            if extra.size else forward_out[1]
                    forward_out = forward_out[0]
                outputs[l:r, ...] = forward_out

        if ret_extra:
            return outputs, extra
        return outputs


    # pretty print
    def __str__(self, prefix=''):
        return prefix + 'FocusCNN: ' \
            'input shape= %s, channel= %d, flat= %d, net_params= %s'%(
            str(self.input_shape), self.input_channel, self.input_shape_flat,
            self.net_params)


# Boosting from multiple neuron network based on change points
class ModelFocusBoost(ModelFocus):
    def __init__(self, cp_detector, *models, **kwargs):
        super(ModelFocusBoost, self).__init__()

        # save models in sequential hierarchy
        self.models = models
        self.cp_detector = cp_detector
        self.train_flags = kwargs.get('train_flags', [False]*len(self.models))

        self.sanitize()


    # push input forward
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

class ModelAttentionCNN(nn.Module):
    def __init__(self, image_shape, net_params):
        super(ModelAttentionCNN, self).__init__()

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


    # push input forward
    def forward(self, img, ret_numpy=False):
        out = img
        for layer in self.layers:
            out = layer(out)
        return out if not ret_numpy else out.detach().numpy()


    # input size of this network
    def input_size(self):
        return (self.input_channel,) + self.input_shape


    # output size of this network
    def output_size(self):
        return self.input_size(self)


    # train with focus model (smoothening)
    def from_focus_model(self, focus_model, dataset, *args, **kwargs):
        print('WARNING: processing whole dataset in one batch!')
        lr = kwargs.get('lr', 1e-3)
        n_iter = kwargs.get('n_iter', 100)

        # get target attention
        frames = dataset.get_frame(0, dataset.n_state)
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames).float()
        focus = focus_model.forward(frames)
        focus_attn = util.focus2attn(focus, self.input_shape)
        focus_attn = 10 * focus_attn - 1  # rescale
        focus_attn = torch.from_numpy(focus_attn).float()

        # train
        lda_1 = 2 # attention regularization
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for t in range(n_iter):
            output = self(frames)
            loss = (output - focus_attn).pow(2).mean() \
                   + (lda_1 + 1) * output.abs().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('iteration %4d: loss= %f'%(t, loss.item()))


    # pretty print
    def __str__(self, prefix=''):
        return prefix + 'AttentionCNN: ' \
            'input shape= %s, channel= %d, flat= %d, net_params= %s'%(
            str(self.input_shape), self.input_channel, self.input_shape_flat,
            self.net_params)