import pickle
import numpy as np
import ChangepointDetection.DynamicsModels as DynamicsModels

def load_from_pickle(pth):
    fid = open(pth, 'rb')
    save_dict = pickle.load(fid)
    fid.close()
    return save_dict

def save_to_pickle(pth, val):
    fid = open(pth, 'wb')
    pickle.dump(val, fid)
    fid.close()

def get_edge(train_edge):
    '''
    edges in format tail1_tail2...->head
    '''
    splt = train_edge.split("->")
    tail = splt[0].split("_")
    head = splt[1]
    return head,tail

def get_cp_models_from_dict(cp_dict):
    keys = [k for k in cp_dict.keys()]
    keys.sort()
    keys.pop(0)
    return keys, [cp_dict[k] for k in keys]

def read_obj_dumps(pth, i= 0, rng=-1):
    '''
    TODO: move this out of this file to support reading object dumps from other sources
    i = -1 means count rng from the back
    rng = -1 means take all after i
    i is start position, rng is number of values
    '''
    obj_dumps = []
    total_len = 0
    if i < 0:
        for line in open(pth + '/object_dumps.txt', 'r'):
            total_len += 1
        print("length", total_len)
        if rng == -1:
            i = 0
        else:
            i = max(total_len - rng, 0)
    current_len = 0
    for line in open(pth + '/object_dumps.txt', 'r'):
        current_len += 1
        if current_len< i:
            continue
        if rng != -1 and current_len > i + rng:
            break
        time_dict = dict()
        for obj in line.split('\t'):
            if obj == "\n":
                continue
            else:
                # print(obj)
                split = obj.split(":")
                name = split[0]
                vals = split[1].split(" ")
                BB = float(vals[0]), float(vals[1])
                # BB = (int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3]))
                # pos = (int(vals[1]),)
                value = (float(vals[2]), )
                time_dict[name] = (BB, value)
        obj_dumps.append(time_dict)
    return obj_dumps


def get_individual_data(name, obj_dumps, pos_val_hash=3):
    '''
    gets data for a particular object, gets everything in obj_dumps
    pos_val_hash gets either position (1), value (2), full position and value (3)
    '''
    data = []
    for time_dict in obj_dumps:
        # print("td1", time_dict[name][1])
        if pos_val_hash == 1:
            data.append(list(time_dict[name][0]))
        elif pos_val_hash == 2:
            # print("td2", list(time_dict[name][1]))
            data.append(list(time_dict[name][1]))
        elif pos_val_hash == 3:
            data.append(list(time_dict[name][0]) + list(time_dict[name][1]))
        else:
            data.append(time_dict[name])
    return np.array(data)