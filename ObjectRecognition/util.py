import os
import numpy as np
import torch
import torch.nn as nn
import warnings
import matplotlib.pyplot as plt
import cv2
from PIL import Image


# find masks of adjacently different entries
def next_noteq(arr):
    return np.array([True] + [not np.equal(arr[i-1], arr[i]) 
                     for i in range(1, arr.shape[0])])

# create directory if not exists
def get_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('created directory', path)
    return path


# extract neighborhood around (focus_x, focus_y)
def extract_neighbor(dataset, focus, idxs, nb_size=(10, 10)):
    focus = (focus * dataset.get_shape()[2:]).astype(int)

    nb_size_2 = (nb_size[0]*2+1, nb_size[1]*2+1)
    neighbors = np.zeros((focus.shape[0],) + nb_size_2)
    pad_size = ((nb_size[0], nb_size[0]), (nb_size[1], nb_size[1]))
    for i, f in enumerate(focus):
        pad_frame = np.pad(dataset.get_frame(idxs[i])[0], pad_size, 'constant')
        f_x, f_y = f[0]+nb_size[0], f[1]+nb_size[1]
        neighbors[i, :] = pad_frame[f_x-nb_size[0]:f_x+nb_size[0]+1,
                                    f_y-nb_size[1]:f_y+nb_size[1]+1]
    return neighbors


# mean square deviation
def mean_square_deviation(xs, ys):
    return np.mean((xs-ys)**2)


# hinged mean deviation
def hinged_mean_square_deviation(xs, ys, alpha_d):
    ALPHA = (alpha_d)**2
    return np.max([np.sum((xs-ys)**2) - ALPHA, 0])


# turn list of event numbers to boolean mask
def list_to_mask(event_list, mask_length):
    event_mask = np.zeros(mask_length, dtype=bool)
    event_mask[event_list] = True
    return event_mask


# find intersection and different of two masks
def match_diffs(cp1, cp2, mask_length):
    mask1 = list_to_mask(cp1, mask_length)
    mask2 = list_to_mask(cp2, mask_length)
    match_mask = mask1 & mask2
    diffs_mask = mask1 ^ mask2

    return match_mask, diffs_mask


# create fake filter
def cheat_init_center(dim, extra, mode):
    warnings.warn('using cheating filter initialization')
    cx = (dim[0]-1)/2
    cy = (dim[1]-1)/2

    if mode == 'paddle':
        invc_filter = np.array([[10.0 if abs(cx-i) < 1 and abs(cy-j) < cy-1 else 
                                -20.0 if abs(cx-i) < 2 and abs(cy-j) < cy else 0
                                 for j in range(dim[0])] for i in range(dim[1])])
    elif mode == 'ball':
        invc_filter = np.array([[10.0 if abs(cx-i) < 1 and abs(cy-j) < 1 else 
                                -20.0 if abs(cx-i) < 2 and abs(cy-j) < 2 else 0
                                 for j in range(dim[0])] for i in range(dim[1])])
    elif mode == 'gaussian':
        c_filter = np.array([[np.sqrt((i-cx)**2 + (j-cy)**2)
                              for j in range(dim[0])] for i in range(dim[1])])
        c_filter /= np.max(c_filter)
        invc_filter = (0.5 - c_filter)*2

    cheated = np.append(invc_filter.flatten(), [0.1]*extra)
    import matplotlib.pyplot as plt; plt.imshow(invc_filter); plt.title('cheating'); plt.show()
    return cheated


# feature normalization by min and max
def feature_normalize(arr):
    mi, mx = np.min(arr), np.max(arr)
    return (arr - mi) / (mx - mi)


# compute confusion matrix metrics
def confuse_metrics(tp, fp, fn, tn):
    if isinstance(tp, list):
        valid = np.logical_or.reduce((tp > 0, fp > 0, fn > 0))
        print(np.sum(valid))
        tp = tp[valid]
        fp = fp[valid]
        fn = fn[valid]
        tn = tn[valid]

    p = tp + fn
    n = fp + tn

    tpr = np.mean(tp / p)
    tnr = np.mean(tn / n)
    ppv = np.mean(tp / (tp + fp))
    npv = np.mean(tn / (tn + fn))
    fnr = np.mean(fn / p)
    fpr = np.mean(fp / n)
    fdr = np.mean(fp / (fp + tp))
    foR = np.mean(fn / (fn + tn))
    acc = np.mean((tp + tn) / (p + n))
    f1 = np.mean((2*tp) / (2*tp + fp + fn))
    mcc = np.mean((tp*tn - fp-fn) / ((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))**0.5)
    bm = np.mean(tpr + tnr - 1)
    mk = np.mean(ppv + npv - 1)

    # print('tpr=', tpr)
    # print('tnr=', tnr)
    # print('ppv=', ppv)
    # print('npv=', npv)
    # print('fnr=', fnr)
    # print('fpr=', fpr)
    # print('fdr=', fdr)
    # print('foR=', foR)
    # print('acc=', acc)
    # print('f1=', f1)
    # print('mcc=', mcc)
    # print('bm=', bm)
    # print('mk=', mk)

    print('tpr= %.3f'%tpr, end=', ')
    print('tnr= %.3f'%tnr, end=', ')
    print('ppv= %.3f'%ppv, end=', ')
    print('npv= %.3f'%npv, end=', ')
    print('fnr= %.3f'%fnr, end=', ')
    print('fpr= %.3f'%fpr, end=', ')
    print('fdr= %.3f'%fdr, end=', ')
    print('foR= %.3f'%foR, end=', ')
    print('acc= %.3f'%acc, end=', ')
    print('f1= %.3f'%f1, end=', ')
    print('mcc= %.3f'%mcc, end=', ')
    print('bm= %.3f'%bm, end=', ')
    print('mk= %.3f'%mk)
    print()


# 2D gaussian pdf centered at (0, 0)
def gaussian_pdf(x, y, sigma, normalized=True):
    coeff = 2*np.pi * (sigma**2) if normalized else 1.0
    return coeff * np.exp(-(x**2 + y**2)/(sigma**2))


# 2D conic function
def conic(x, y, d=None):
    return (x**2 + y**2)**0.5


# 2D step function within distance
def step2d_fn(x, y, dist):
    return np.where(np.sqrt(x**2 + y**2) < dist, 1.0, 0.0)


# convert from focus to frame intensity in [0.0, 1.0]
def focus2attn(focus, input_shape, d=0.02, fn=gaussian_pdf):
    attn = np.zeros((focus.shape[0], 1) + input_shape)
    for i, f in enumerate(focus):
        xs = np.linspace(0, 1, input_shape[0], endpoint=False) - f[0]
        ys = np.linspace(0, 1, input_shape[1], endpoint=False) - f[1]
        attn[i, ...] = fn(xs[:, None], ys[None, :], d)
    return attn # / np.max(attn, axis=0)


# torch softmax
def nn_logsoftmax(img):
    img_size = img.size()
    return nn.LogSoftmax()(img.view(img_size[0], -1)).view(*img_size)


# pick first element in list
def pick_first(x):
    return x[0]


# save image
def imsave_bg(save_path, img):
    bg = np.array(Image.open("frame_mean.png"))
    bg = 255.0 * bg[:, :, None] * np.ones(4)[None, None, :] / bg.max()
    bg[:, :, 3] = 255.0
    bg = 255.0 * bg / bg.max()
    bg_p = 0.3
    import matplotlib.pyplot as plt
    img =  plt.cm.get_cmap('inferno')(img)
    rescaled = (255.0 / img.max() * (img - img.min()))
    rescaled[:, :, 3] = 255.0
    rescaled = (1.0 - bg_p) * rescaled + bg_p * bg
    im = Image.fromarray(rescaled.astype(np.uint8))
    im.save(save_path)


# save image
def imsave(save_path, img):
    rescaled = (255.0 / img.max() * (img - img.min()))
    im = Image.fromarray(rescaled.astype(np.uint8))
    im.save(save_path)


# save image of skew counting
def count_imsave(save_path, cnt, cm=np.ones(3)):
    cnt = np.sqrt(1.0 - (1 - cnt / np.max(cnt))**2)
    imsave(save_path, cnt[:, :, None] * cm[None, None, :])


# binarize frames
def binarize(frames, binarize):
    frames[frames < binarize] = 0.0
    frames[frames >= binarize] = 1.0
    return frames


"""
Image-Focus Augmentation Function
"""

# no-op function on image
def noop_x(imgs, focus):
    return imgs


# no-op function on focus
def noop_y(imgs, focus):
    return focus


# remove by mean
def remove_mean(imgs, focus, nb_size=(4, 9)):
    in_np = isinstance(imgs, np.ndarray)
    if not in_np:
        imgs = imgs.detach().numpy()
    imgs = np.array(imgs)
    focus = (focus * imgs.shape[2:]).astype(int)

    # get neighborhoods
# get mean of neighborhood around focus
def image_focus_mean(imgs, focus, nb_size):
    all_mean = np.mean(imgs, axis=0).reshape(imgs.shape[-2:])
    nb_size_2 = (nb_size[0]*2+1, nb_size[1]*2+1)
    neighbors = np.zeros((focus.shape[0],) + nb_size_2)
    pad_size = ((nb_size[0], nb_size[0]), (nb_size[1], nb_size[1]))
    for i, f in enumerate(focus):
        pad_frame = np.pad(imgs[i][0] - all_mean, pad_size, 'constant')
        f_x, f_y = f[0]+nb_size[0], f[1]+nb_size[1]
        neighbors[i, :] = pad_frame[f_x-nb_size[0]:f_x+nb_size[0]+1,
                                    f_y-nb_size[1]:f_y+nb_size[1]+1]
    focus_mean = np.mean(neighbors, axis=0)

    EPS = 0.01
    focus_mean[focus_mean <= EPS] = 0.0
    focus_mean[focus_mean > EPS] = 1.0
    return focus_mean


# subtract image from image batch
def image_focus_subtract(imgs, focus, focus_mean, nb_size):
    imgs = np.array(imgs)
    pad_size = ((nb_size[0], nb_size[0]), (nb_size[1], nb_size[1]))
    for i, f in enumerate(focus):
        pad_frame = np.pad(imgs[i][0], pad_size, 'constant')
        f_x, f_y = f[0]+nb_size[0], f[1]+nb_size[1]
        pad_frame[f_x-nb_size[0]:f_x+nb_size[0]+1, \
                  f_y-nb_size[1]:f_y+nb_size[1]+1] -= focus_mean
        pad_frame[pad_frame < 0] = 0.0
        imgs[i, 0, ...] = pad_frame[pad_size[0][0]:-pad_size[0][1], \
                                    pad_size[1][0]:-pad_size[1][1]]
    # for f, img in zip(focus, imgs):
    #     print(f)
    #     img = img[::]
    #     img[:, f[0], :] = 255
    #     img[:, :, f[1]] = 255
    #     plt.imshow(np.squeeze(img))
    #     plt.show()
        # cv2.imshow("frame", np.squeeze(img))
        # cv2.waitKey(30)
    return imgs


# remove by mean of the batch
def remove_mean_batch(imgs, focus, nb_size=(5, 5)):
    in_np = isinstance(imgs, np.ndarray)
    if not in_np:
        imgs = imgs.detach().numpy()
    focus = (focus * imgs.shape[2:]).astype(int)
    focus_mean = image_focus_mean(imgs, focus, nb_size)
    imgs = image_focus_subtract(imgs, focus, focus_mean, nb_size)
    return imgs if in_np else torch.from_numpy(imgs).float()


# remove by mean, memorized mean
class RemoveMeanMemory:

    def __init__(self, nb_size):
        self.nb_size = nb_size
        self.n_mean = 0
        self.mean = np.zeros((nb_size[0]*2+1, nb_size[1]*2+1))


    # remove this batch with memorized mean and update it
    def remove_mean_memory(self, imgs, focus):
        in_np = isinstance(imgs, np.ndarray)
        if not in_np:
            imgs = imgs.detach().numpy()
        focus = (focus * imgs.shape[2:]).astype(int)
        focus_mean = image_focus_mean(imgs, focus, self.nb_size)
        self.mean = (self.n_mean * self.mean + focus.shape[0] * focus_mean) \
                    / (self.n_mean + focus.shape[0])
        self.n_mean = self.n_mean + focus.shape[0]
        imgs = image_focus_subtract(imgs, focus, self.mean, self.nb_size)
        return imgs if in_np else torch.from_numpy(imgs).float()


    # call routine
    def __call__(self, imgs, focus):
        return self.remove_mean_memory(imgs, focus)


"""
Jump protection
"""

# augmented focus position for jumping
class JumpAugmentFocus:

    def __init__(self, focus):
        self.v = None
        self.conf_focus = np.array(focus)
        self.pred_focus = np.array(focus)
        self.t = 1


    # update with new focus
    def update(self, next_focus):
        self.v = np.array(next_focus) - self.conf_focus  # minus conf_focus or pred_focus?
        self.conf_focus = np.array(next_focus)
        self.pred_focus = np.array(next_focus)
        self.t += 1


    # step with predicted velocity
    def step(self):
        self.pred_focus += self.v
        self.t += 1


    # expected distance error
    def dist(self, next_focus):
        return np.sum((next_focus - self.conf_focus)**2)**0.5


# jump filtering based on current and expected positions
class JumpFiltering:

    def __init__(self, jump_count_t, jump_threshold):
        self.jump_count_t = jump_count_t
        self.jump_threshold = jump_threshold
        self.cur_pos = None
        self.jump_pos = None


    # remove this batch with memorized mean and update it
    def jump_filter(self, imgs, focus):
        # if first time
        if self.cur_pos is None:
            self.cur_pos = JumpAugmentFocus(focus)
            self.jump_pos = None
            return focus

        # if next time...
        if self.cur_pos.dist(focus) <= self.jump_threshold:
            # non-jump, it's okay
            self.cur_pos.update(focus)
            self.jump_pos = None
        else:
            # jump, update jump_pos
            if self.jump_pos is None:
                self.jump_pos = JumpAugmentFocus(focus)
            else:
                if self.jump_pos.dist(focus) <= self.jump_threshold:
                    self.jump_pos.update(focus)
                else:
                    self.jump_pos = JumpAugmentFocus(focus)

            # check if jumping is valid
            if self.jump_pos.t >= self.jump_count_t:
                self.cur_pos = self.jump_pos
                self.jump_pos = None
            else:
                self.cur_pos.step()

        return np.clip(self.cur_pos.pred_focus, 0.0, 83.0/84.0)  # TODO: eps?


    # call routine
    def __call__(self, imgs, focus):
        assert 0 <= len(focus.shape) <= 2
        if len(focus.shape) == 1:
            new_focus = self.jump_filter(imgs, focus)
        if len(focus.shape) == 2:
            new_focus = np.zeros(focus.shape)
            for i in range(focus.shape[0]):
                new_focus[i] = self.jump_filter(imgs, focus[i])
        return new_focus