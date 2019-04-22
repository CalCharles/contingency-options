import pickle, os
import numpy as np
import ChangepointDetection.DynamicsModels as DynamicsModels

def load_from_pickle(pth):
    fid = open(pth, 'rb')
    save_dict = pickle.load(fid)
    fid.close()
    return save_dict

def save_to_pickle(pth, val):
    try:
        os.makedirs(os.path.join(*pth.split("/")[:-1]))
    except OSError:
        pass
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
    if len(obj_dumps) > 0:
        names = list(obj_dumps[0].keys())
        relevant_names = [n for n in names if n.find(name) != -1]
        relevant_names.sort() # sorting procedure should be fixed between this and state getting
    for time_dict in obj_dumps:
        # print("td1", time_dict[name][1])
        if pos_val_hash == 1:
            data.append(sum([list(time_dict[name][0]) for name in relevant_names], []))
        elif pos_val_hash == 2:
            # print("td2", list(time_dict[name][1]))
            data.append(sum([list(time_dict[name][1]) for name in relevant_names], []))
        elif pos_val_hash == 3:
            data.append(sum([list(time_dict[name][0]) + list(time_dict[name][1]) for name in relevant_names], []))
        else:
            data.append(sum([list(time_dict[name]) for name in relevant_names], []))
    return np.array(data)

def default_value_arg(kwargs, key, value):
    if key in kwargs:
        return kwargs[key]
    else:
        return value

def render_dump(obj_dumps):
    frame = np.zeros((84,84), dtype = 'uint8')
    for bn in [bn for bn in obj_dumps.keys() if bn.find('Block') != -1]:
        block = obj_dumps[bn]
        pos = (int(block[0][0]), int(block[0][1]))
        # print(pos, block[1])
        if block[1][0] == 1:
            frame[pos[0]-1:pos[0]+1, pos[1]-1:pos[1]+2] = .5 * 255
    walln = "TopWall"
    wall = obj_dumps[walln]
    pos = (int(wall[0][0]), int(wall[0][1]))
    # print(pos)
    width = 84
    height = 4
    frame[pos[0]-height//2:pos[0]+height//2, pos[1]-width//2:pos[1]+width//2] = .3 * 255

    walln = "RightSideWall"
    wall = obj_dumps[walln]
    pos = (int(wall[0][0]), int(wall[0][1]))
    width = 4
    height = 84
    frame[pos[0]-height//2:pos[0]+height//2, pos[1]-width//2:pos[1]+width//2] = .3 * 255

    walln = "LeftSideWall"
    wall = obj_dumps[walln]
    pos = (int(wall[0][0]), int(wall[0][1]))
    width = 4
    height = 84
    frame[pos[0]-height//2:pos[0]+height//2, pos[1]-width//2:pos[1]+width//2] = .3 * 255

    walln = "BottomWall"
    wall = obj_dumps[walln]
    pos = (int(wall[0][0]), int(wall[0][1]))
    width = 84
    height = 4
    frame[pos[0]-height//2:pos[0]+height//2, pos[1]-width//2:pos[1]+width//2] = .3 * 255

    pos = (int(obj_dumps["Paddle"][0][0]), int(obj_dumps["Paddle"][0][1]))
    width = 7
    height = 2
    frame[pos[0]:pos[0]+height, pos[1]-3:pos[1]+4] = .75 * 255

    pos = (int(obj_dumps["Ball"][0][0]), int(obj_dumps["Ball"][0][1]))
    width = 2
    height = 2
    frame[pos[0]-1:pos[0]+1, pos[1]-1:pos[1]+1] = 1.0 * 255
    return frame
