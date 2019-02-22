import os
import numpy as np
import warnings


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
def hinged_mean_square_deviation(xs, ys):
    ALPHA = (0.1)**2
    return np.max([np.mean((xs-ys)**2) - ALPHA, 0])


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