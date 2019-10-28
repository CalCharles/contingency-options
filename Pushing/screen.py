# screen.py
import numpy as np
import cv2
import argparse
import sys
sys.path.insert(0, "/home/calcharles/research/contingency-options/")
from Pushing.objects import *
import imageio as imio
import os, copy
from Environments.environment_specification import RawEnvironment

class Pushing(RawEnvironment):
    def __init__(self, pushgripper=False, frameskip=5):
        super().__init__()
        self.actionObj = Action()
        self.gripper = CartesianGripper(2)
        self.stick = Stick(2)
        # self.ball = Sphere(2)
        self.ball = Block(2)
        self.target = Target(2)
        if pushgripper:
            self.gripper = CartesianPusher(2)
            self.objects = [self.actionObj, self.gripper, self.ball, self.target]
        else:
            self.objects = [self.actionObj, self.gripper, self.stick, self.ball, self.target]
        self.num_actions = 5
        self.render()
        self.reset()
        self.reset_counter = 0
        self.reward = -.01
        self.frameskip = frameskip
        self.distance_reward = False
        self.name = "Pusher"

    def use_distance_reward(self):
        self.distance_reward = True

    def reset_object(self, o):
        newy = o.limits[0] + np.round((o.limits[2] - o.limits[0]) * np.random.rand())
        newx = o.limits[1] + np.round((o.limits[3] - o.limits[1]) * np.random.rand())
        o.updateBounding(np.array([newy, newx]))
        o.touched = False
        o.gripped = None


    def reset(self):
        for o in self.objects:
            self.reset_object(o)
        self.reset_counter = 0

    def render(self):
        frame = np.zeros((84,84))
        for i, o in enumerate(self.objects):
            if o.isAABB:
                # frame[int(o.center[0] + np.random.randint(12) - 6), :] = 1.0
                # frame[:, int(o.center[1] + np.random.randint(12) - 6)] = 1.0
                # print(type(o), o.bb)
                by, ey = max(int(np.round(o.bb[0])), 0), min(int(np.round(o.bb[2])), 84)
                bx, ex = max(int(np.round(o.bb[1])), 0), min(int(np.round(o.bb[3])), 84)
                frame[by:ey, bx:ex] = 1.0 / (len(self.objects)-1) * i
                if o.isGripper:
                    # yd = int(np.round(o.bb[2])) - int(np.round(o.bb[0]))
                    # xd = int(np.round(o.bb[3])) - int(np.round(o.bb[1]))
                    yd = int(np.round(o.bb[2])) - int(np.round(o.bb[0]))
                    xd = int(np.round(o.bb[3])) - int(np.round(o.bb[1]))
                    frame[int(np.round(o.bb[0])):int(np.round(o.bb[2]))-int(np.round(yd * .3)), int(np.round(o.bb[1]))+int(np.round(xd * .3)):int(np.round(o.bb[3]))-int(np.round(xd * .3))] = 0.0
                    # print(int(np.round(o.bb[0])),int(np.round(o.bb[2]))-int(np.round(yd * .3)), int(np.round(o.bb[1]))+int(np.round(xd * .3)),int(np.round(o.bb[3]))-int(np.round(xd * .3)))
                if type(o) == Stick:
                    by, ey = max(int(np.round(o.bb[0])), 0), min(int(np.round(o.bb[2])), 84)
                    bx, ex = max(int(np.round(o.bb[1])) + 1, 0), min(int(np.round(o.bb[3])) - 1, 84)
                    # print(bx, ex, o.bb)
                    frame[by:ey, bx:ex] = .7

            elif o.isSphere:
                Y,X = np.ogrid[:84,:84]
                d = np.sqrt((X-o.center[1])**2 + (Y-o.center[0])**2)
                mask = d <= o.radius
                frame[mask] = 1.0 / (len(self.objects)-1) * i
            # print(1.0 / (len(self.objects)) * (i + 1))
        self.frame = frame
        return self.frame
    
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
            # self.obj_rec[i].append([obj.name, obj.pos, obj.attribute])
            object_dumps.write(obj.name + ":" + " ".join(map(str, obj.getMidpoint())) + " " + str(float(obj.attribute)) + "\t") # TODO: attributes are limited to single floats
        object_dumps.write("\n") # TODO: recycling does not stop object dumping
        imio.imsave(os.path.join(state_path, "state" + str(count % 2000) + ".png"), (self.frame * 255.0).astype('uint8'))
        if len(self.all_dir) > 0:
            state_path = os.path.join(save_path, self.all_dir)
            try:
                os.makedirs(state_path)
            except OSError:
                pass
            imio.imsave(os.path.join(state_path, "state" + str(count) + ".png"), (self.frame * 255.0).astype('uint8'))



    def step(self, action, render=True, frameskip=2):
        if self.reset_counter == 1:
            self.reward = 0
        for i in range(self.frameskip):
            self.actionObj.attribute = action
            self.gripper.applyAction(self.actionObj)
            if len(self.save_path) != 0 and i == self.frameskip-1:
                if self.itr != 0:
                    object_dumps = open(os.path.join(self.save_path, "object_dumps.txt"), 'a')
                else:
                    object_dumps = open(os.path.join(self.save_path, "object_dumps.txt"), 'w') # create file if it does not exist
                self.write_objects(object_dumps, self.save_path)
                object_dumps.close()

            for i in range(len(self.objects)):
                for j in range(len(self.objects)): # objects in order
                    if i != j: # no self-interactions yet
                        o1, o2 = self.objects[i], self.objects[j]
                        contact = o1.getContact(o2)
                        o1.actContact(contact, o2)
            for i in range(len(self.objects)):
                self.objects[i].move()
            if self.ball.center[0] < 0 or self.ball.center[0] > 84 or self.ball.center[1] < 0 or self.ball.center[1] > 84:
                self.reset_object(self.ball)  
            extracted_state = {obj.name: (obj.getMidpoint(), (obj.getAttribute(), )) for obj in self.objects}
            rawframe = None
            if render:
                rawframe = self.render()
            done = False
            if self.distance_reward:
                self.reward = -.0001 * np.linalg.norm(self.ball.center - self.target.center, 1) + -.03 * (1-int(self.ball.moved))
                # print(self.ball.center, int(self.ball.moved), self.reward)
            if self.target.touched:
                self.reward = 1
                self.reset()
                done = True
                break
            if self.reset_counter % 300 == 0 and self.reset_counter > 0:
                self.reset()
                done = True
                break
        # print(self.reward)
        self.reset_counter += 1
        self.itr += 1
        return self.frame, extracted_state, done

    def getState(self, render=True):
        extracted_state = {obj.name: (obj.getMidpoint(), (obj.getAttribute(), )) for obj in self.objects}
        rawframe = None
        if render:
            rawframe = self.render()
        return rawframe, extracted_state

class RandomPolicy():
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, screen):
        return np.random.randint(self.action_space)

class RandomRepeatPolicy():
    def __init__(self, action_space, repeat=3):
        self.action_space = action_space
        self.repeat = repeat
        self.cnt = 0
        self.last_action = 0

    def act(self, screen):
        if self.cnt == 0:
            self.last_action = np.random.randint(self.action_space)
            self.cnt = self.repeat
        self.cnt -= 1
        return self.last_action

def demonstrate(save_dir, num_frames, pushgripper=False):
    pushingDomain = Pushing(pushgripper)
    quit = False
    pushingDomain.set_save(0, save_dir, 0)

    for i in range(num_frames):
        frame = pushingDomain.render()
        cv2.imshow('frame',frame)
        action = 0
        key = cv2.waitKey(500)
        if key == ord('q'):
            quit = True
        elif key == ord('a'):
            action = 3
        elif key == ord('w'):
            action = 1
        elif key == ord('s'):
            action = 2
        elif key == ord('d'):
            action = 4
        pushingDomain.step(action)

def run(save_dir, num_frames, policy, pushgripper=False):
    # running with a random policy
    pushingDomain = Pushing(pushgripper, frameskip=3)
    pushingDomain.set_save(0, save_dir, 0)
    for i in range(num_frames):
        action = policy.act(pushingDomain)
        pushingDomain.step(action)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train object recognition')
    parser.add_argument('savedir',
                        help='base directory to save results')
    parser.add_argument('--num-frames', type=int, default=1000,
                        help='number of frames to run')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='number of training iterations')
    parser.add_argument('--demonstrate', action='store_true', default=False,
                        help='get the data from demonstrations or from random motion')
    parser.add_argument('--pushgripper', action='store_true', default=False,
                        help='run the pushing gripper domain')
    args = parser.parse_args()
    if args.demonstrate:
        demonstrate(args.savedir, args.num_frames, args.pushgripper)
    else:
        # run(args.savedir, args.num_frames, RandomPolicy(5), args.pushgripper)
        run(args.savedir, args.num_frames, RandomRepeatPolicy(5), args.pushgripper)
