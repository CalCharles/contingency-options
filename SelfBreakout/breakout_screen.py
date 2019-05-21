# Screen
import sys
sys.path.insert(0, "/home/calcharles/research/contingency-options/")
import numpy as np
from SelfBreakout.breakout_objects import *
import imageio as imio
import os, copy
from Environments.environment_specification import RawEnvironment

class Screen(RawEnvironment):
    def __init__(self, frameskip = 1):
        super(Screen, self).__init__()
        self.reset()
        self.num_actions = 4
        self.average_points_per_life = 0
        self.itr = 0
        self.save_path = ""
        self.recycle = -1
        self.frameskip = frameskip
        self.total_score = 0

    def reset(self):
        vel= np.array([np.random.randint(1,2), np.random.choice([-1,1])])
        # self.ball = Ball(np.array([52, np.random.randint(14, 70)]), 1, vel)
        self.ball = Ball(np.array([46, np.random.randint(20, 36)]), 1, vel)
        self.paddle = Paddle(np.array([71, 84//2]), 1, np.zeros((2,), dtype = np.int64))
        self.actions = Action(np.zeros((2,), dtype = np.int64), 0)
        self.reward = 0
        self.blocks = []
        for i in range(5):
            for j in range(20):
                self.blocks.append(Block(np.array([22 + i * 2,12 + j * 3]), 1, i * 20 + j))
                # self.blocks.append(Block(np.array([32 + i * 2,12 + j * 3]), 1, i * 20 + j))

        self.walls = []
        # Topwall
        self.walls.append(Wall(np.array([4, 4]), 1, "Top"))
        self.walls.append(Wall(np.array([80, 4]), 1, "Bottom"))
        self.walls.append(Wall(np.array([0, 8]), 1, "LeftSide"))
        self.walls.append(Wall(np.array([0, 72]), 1, "RightSide"))
        self.animate = [self.ball, self.paddle]
        self.objects = [self.ball, self.paddle, self.actions] + self.blocks + self.walls
        self.obj_rec = [[] for i in range(len(self.objects))]
        self.counter = 0

        self.render_frame()

    def render_frame(self):
        self.frame = np.zeros((84,84), dtype = 'uint8')
        for block in self.blocks:
            if block.attribute == 1:
                self.frame[block.pos[0]:block.pos[0]+block.height, block.pos[1]:block.pos[1]+block.width] = .5 * 255
        for wall in self.walls:
            self.frame[wall.pos[0]:wall.pos[0]+wall.height, wall.pos[1]:wall.pos[1]+wall.width] = .3 * 255
        ball, paddle = self.ball, self.paddle
        self.frame[ball.pos[0]:ball.pos[0]+ball.height, ball.pos[1]:ball.pos[1]+ball.width] = 1 * 255
        self.frame[paddle.pos[0]:paddle.pos[0]+paddle.height, paddle.pos[1]:paddle.pos[1]+paddle.width] = .75 * 255

    def extracted_state(self):
        return np.array([obj.getMidpoint() for obj in self.objects], dtype=np.float64)

    def get_num_points(self):
        total = 0
        for block in self.blocks:
            if block.attribute == 0:
                total += 1
        # print(total)
        return total

    def extracted_state_dict(self):
        return {obj.name: obj.getMidpoint() for obj in self.objects}

    def getState(self):
        self.render_frame()
        return self.frame, {obj.name: (obj.getMidpoint(), (obj.getAttribute(), )) for obj in self.objects}

    def step(self, action, render=True): # TODO: remove render as an input variable
        done = False
        last_loss = self.ball.losses
        self.reward = 0
        for i in range(self.frameskip):
            self.actions.take_action(action[0])
            if len(self.save_path) != 0 and i == 0:
                if self.itr != 0:
                    object_dumps = open(os.path.join(self.save_path, "object_dumps.txt"), 'a')
                else:
                    object_dumps = open(os.path.join(self.save_path, "object_dumps.txt"), 'w') # create file if it does not exist
                self.write_objects(object_dumps, self.save_path)
                object_dumps.close()
            for obj1 in self.animate:
                for obj2 in self.objects:
                    if obj1.name == obj2.name:
                        continue
                    else:
                        preattr = obj2.attribute
                        obj1.interact(obj2)
                        if preattr != obj2.attribute:
                            self.reward += 1
                            self.total_score += 1
            for ani_obj in self.animate:
                ani_obj.move()
            extracted_state = {obj.name: (obj.getMidpoint(), (obj.getAttribute(), )) for obj in self.objects}
            if last_loss != self.ball.losses:
                done = True
            if self.ball.losses == 5:
                self.average_points_per_life = self.total_score / 5.0
                done = True
                self.episode_rewards.append(self.total_score)
                self.total_score = 0
                self.reset()
            if self.itr % 100 == 0 and self.get_num_points() == len(self.blocks):
                self.reset()

            if render:
                self.render_frame()
        self.itr += 1
        return self.frame, extracted_state, done

    def write_objects(self, object_dumps, save_path):
        if self.recycle > 0:
            state_path = os.path.join(save_path, str((self.itr % self.recycle)//2000))
            count = self.itr % self.recycle
        else:
            state_path = os.path.join(save_path, str(self.itr//2000))
            count = self.itr
        try:
            os.makedirs(state_path)
        except OSError:
            pass
        for i, obj in enumerate(self.objects):
            self.obj_rec[i].append([obj.name, obj.pos, obj.attribute])
            object_dumps.write(obj.name + ":" + " ".join(map(str, obj.getMidpoint())) + " " + str(float(obj.attribute)) + "\t") # TODO: attributes are limited to single floats
        object_dumps.write("\n") # TODO: recycling does not stop object dumping
        imio.imsave(os.path.join(state_path, "state" + str(count % 2000) + ".png"), self.frame)
        if len(self.all_dir) > 0:
            state_path = os.path.join(save_path, self.all_dir)
            try:
                os.makedirs(state_path)
            except OSError:
                pass
            imio.imsave(os.path.join(state_path, "state" + str(count) + ".png"), self.frame)


    def run(self, policy, iterations = 10000, render=False, save_path = "runs/", duplicate_actions=1):
        self.save_path = save_path
        self.itr = 0
        try:
            os.makedirs(save_path)
        except OSError:
            pass
        object_dumps = open(os.path.join(save_path,"object_dumps.txt"), 'w')
        if render:
            self.render_frame()
        for self.itr in range(iterations):
            if self.itr % duplicate_actions == 0:
                action = policy.act(self)
                last_action = action
            else:
                action = last_action
            self.actions.take_action(action)
            self.write_objects(object_dumps, save_path)


            for obj1 in self.animate:
                for obj2 in self.objects:
                    if obj1.name == obj2.name:
                        continue
                    else:
                        obj1.interact(obj2)
            for ani_obj in self.animate:
                ani_obj.move()
            if self.ball.losses == 5:
                self.reset()
            if self.itr % 100 == 0 and self.get_num_points() == len(self.blocks):
                self.reset()
            if render:
                self.render_frame()
        object_dumps.close()

class Policy():
    def act(self, screen):
        print ("not implemented")

class RandomPolicy(Policy):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, screen):
        return np.random.randint(self.action_space)

class RandomConsistentPolicy(Policy):
    def __init__(self, action_space, change_prob):
        self.action_space = action_space
        self.change_prob = change_prob
        self.current_action = np.random.randint(self.action_space)

    def act(self, screen):
        if np.random.rand() < self.change_prob:
            self.current_action = np.random.randint(self.action_space)
        return self.current_action

class RotatePolicy(Policy):
    def __init__(self, action_space, hold_count):
        self.action_space = action_space
        self.hold_count = hold_count
        self.current_action = 0
        self.current_count = 0

    def act(self, screen):
        self.current_count += 1
        if self.current_count >= self.hold_count:
            self.current_action = np.random.randint(self.action_space)
            # self.current_action = (self.current_action+1) % self.action_space
            self.current_count = 0
        return self.current_action

class BouncePolicy(Policy):
    def __init__(self, action_space):
        self.action_space = action_space
        self.internal_screen = Screen()
        self.objective_location = 84//2
        self.last_paddlehits = -1

    def act(self, screen):
        # print(screen.ball.paddlehits, screen.ball.losses, self.last_paddlehits)
        if screen.ball.paddlehits + screen.ball.losses > self.last_paddlehits or (screen.ball.paddlehits + screen.ball.losses == 0 and self.last_paddlehits != 0):
            if (screen.ball.paddlehits + screen.ball.losses == 0 and self.last_paddlehits != 0):
                self.last_paddlehits = 0
            self.internal_screen = copy.deepcopy(screen)
            # print(self.internal_screen.ball.pos, screen.ball.pos, self.last_paddlehits)
            while self.internal_screen.ball.pos[0] < 71:
                self.internal_screen.step([0])
            # print(self.internal_screen.ball.pos, screen.ball.pos, self.last_paddlehits)
            self.objective_location = self.internal_screen.ball.pos[1] + np.random.choice([-1, 0, 1])
            self.last_paddlehits += 1
        if self.objective_location > screen.paddle.getMidpoint()[1]:
            return 3
        elif self.objective_location < screen.paddle.getMidpoint()[1]:
            return 2
        else:
            return 0

def abbreviate_obj_dump_file(pth, new_path, get_last=-1):
    total_len = 0
    if get_last > 0:
        for line in open(os.path.join(pth,  'object_dumps.txt'), 'r'):
            total_len += 1
    current_len = 0
    new_file = open(os.path.join(new_path, 'object_dumps.txt'), 'w')
    for line in open(os.path.join(pth,  'object_dumps.txt'), 'r'):
        current_len += 1
        if current_len< total_len-get_last:
            continue
        new_file.write(line)
    new_file.close()

def read_obj_dumps(pth, i= 0, rng=-1, dumps_name='object_dumps.txt'):
    '''
    TODO: move this out of this file to support reading object dumps from other sources
    i = -1 means count rng from the back
    rng = -1 means take all after i
    i is start position, rng is number of values
    '''
    obj_dumps = []
    total_len = 0
    if i < 0:
        for line in open(os.path.join(pth, dumps_name), 'r'):
            total_len += 1
        print("length", total_len)
        if rng == -1:
            i = 0
        else:
            i = max(total_len - rng, 0)
    current_len = 0
    for line in open(os.path.join(pth, dumps_name), 'r'):
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

def get_raw_data(pth, i=0, rng=-1):
    '''
    loads raw frames, i denotes starting position, rng denotes range of values. If 
    '''
    frames = []
    if rng == -1:
        try:
            f = i
            while True:
                frames.append(imio.load(os.path.join(pth, "state" + str(f) + ".png")))
                f += 1
        except OSError as e:
            return frames
    else:
        for f in range(i, i + rng[1]):
            frames.append(imio.load(os.path.join(pth, "state" + str(f) + ".png")))
    return frames

def get_action_from_dump(obj_dumps):
    return int(obj_dumps["Action"][1])

def get_individual_data(name, obj_dumps, pos_val_hash=3):
    '''
    gets data for a particular object, gets everything in obj_dumps
    pos_val_hash gets either position (1), value (2), full position and value (3)
    '''
    data = []
    for time_dict in obj_dumps:
        if pos_val_hash == 1:
            data.append(time_dict[name][0])
        elif pos_val_hash == 2:
            data.append(time_dict[name][1])
        elif pos_val_hash == 3:
            data.append(list(time_dict[name][0]) + list(time_dict[name][1]))
        else:
            data.append(time_dict[name])
    return data

def hot_actions(action_data):
    for i in range(len(action_data)):
        hot = np.zeros(4)
        hot[int(action_data[i])] = 1
        action_data[i] = hot.tolist()
    return action_data


if __name__ == '__main__':
    screen = Screen()
    # policy = RandomPolicy(4)
    policy = RotatePolicy(4, 7)
    # policy = BouncePolicy(4)
    screen.run(policy, render=True, iterations = 1000, duplicate_actions=1, save_path=sys.argv[1])