import numpy as np
import torch, cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from Models.models import pytorch_model, Model
import matplotlib.pyplot as plt


class ImageModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        args, num_inputs, num_outputs, factor = self.get_args(kwargs)
        # TODO: assumes images of size 84x84, make general
        self.no_preamble = True
        self.num_stack = args.num_stack
        self.factor = factor
        self.conv1 = nn.Conv2d(self.num_stack, 2 * factor, 8, stride=4)
        self.conv2 = nn.Conv2d(2 * factor, 4 * factor, 4, stride=2)
        self.conv3 = nn.Conv2d(4 * factor, 2 * factor, 3, stride=1)
        self.viewsize = 7
        if args.post_transform_form == 'none':
            self.linear1 = None
            self.insize = 2 * self.factor * self.viewsize * self.viewsize
            self.init_last(num_outputs)
        else:
            self.linear1 = nn.Linear(2 * factor * self.viewsize * self.viewsize, self.insize)
            self.layers.append(self.linear1)
        self.factor = args.factor
        self.layers.append(self.conv1)
        self.layers.append(self.conv2)
        self.layers.append(self.conv3)
        self.reset_parameters()

    def hidden(self, inputs, resp):
        norm_term = 1.0
        if self.use_normalize:
            norm_term =  255.0
        # print(inputs.shape)
        # print(x.sum())
        x = self.conv1(inputs / norm_term)
        # print(x.sum())
        x = self.acti(x)

        x = self.conv2(x)
        # print(x.sum())
        x = self.acti(x)

        x = self.conv3(x)
        # print(x.sum())
        x = self.acti(x)
        # print(x)
        x = x.view(-1, 2 * self.factor * self.viewsize * self.viewsize)
        # print(x.sum())
        x = self.acti(x)
        if self.linear1 is not None:
            x = self.linear1(x)
            x = self.acti(x)
        return x

    # def forward(self, inputs, resp):
    #     x = self.hidden(inputs, resp)
    #     values, dist_entropy, probs, Q_vals = super().forward(x)
    #     return values, dist_entropy, probs, Q_vals

class RawModel(ImageModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # args, num_inputs, num_outputs, factor = self.get_args(kwargs)
        # # TODO: assumes images of size 84x84, make general
        # self.conv1 = nn.Conv2d(1, 2 * factor, 8, stride=4)
        # self.conv2 = nn.Conv2d(2 * factor, 4 * factor, 4, stride=2)
        # self.conv3 = nn.Conv2d(4 * factor, 8 * factor, 3, stride=1)
        # self.viewsize = 7
        # self.linear1 = nn.Linear(8 * factor * self.viewsize * self.viewsize, self.insize)
        # self.factor = args.factor
        # self.layers.append(self.conv1)
        # self.layers.append(self.conv2)
        # self.layers.append(self.conv3)
        # self.layers.append(self.linear1)
        # self.reset_parameters()

    def hidden(self, inputs, resp):
        # print(inputs.shape)
        inputs = inputs.view(-1, self.num_stack, 84, 84)
        # cv2.imshow('frame',pytorch_model.unwrap(inputs[0][0]))
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     pass

        x = super().hidden(inputs, resp)
        return x


class ObjectSumImageModel(ImageModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        args, num_inputs, num_outputs, factor = self.get_args(kwargs)
        # TODO: assumes images of size 84x84
        # TODO: only handles bounds as input, and no object shape. If useful, we would need both
        # TODO: valid input orders: 83, 75, 67, 59, 51, 43, 35
        self.scale = args.scale
        self.period = args.period
        self.order =  args.order + 1# num population repurposed for tile factor
        self.order_vector = [] # shape: [self.order ** 2, 2]
        for j in range(self.order):
            for k in range(self.order):
                self.order_vector.append([j / (self.order - 1), k / (self.order - 1)])
        self.order_vector = pytorch_model.wrap(self.order_vector, cuda = args.cuda).detach()

        self.viewsize = int((((self.order-4)/4-2)/2-2))
        print("insize", self.insize)
        self.conv1 = nn.Conv2d(1, 2 * factor, 8, stride=4)
        self.conv2 = nn.Conv2d(2 * factor, 4 * factor, 4, stride=2)
        self.conv3 = nn.Conv2d(4 * factor, 8 * factor, 3, stride=1)
        self.linear1 = nn.Linear(8 * factor * self.viewsize * self.viewsize, self.insize)
        self.layers[-4] = self.conv1
        self.layers[-3] = self.conv2
        self.layers[-2] = self.conv3
        self.layers[-1] = self.linear1
        self.reset_parameters()

    def normalize(self, inputs):
        return inputs / 84 # assuming 84 size images

    def create_image(self, inputs): # currently, the stack is essentially a motion blurred image
        # print(inputs.shape)
        # inputs = self.normalize(inputs)
        batch = []
        for dpt in inputs:
            image = torch.zeros(self.order ** 2)
            if self.iscuda:
                image = image.cuda()
            # print (image.shape)
            for i in range(self.num_inputs // 2):
                # print((torch.exp(-(self.order_vector - dpt[i*2:(i+1)*2]).pow(2).sum(dim=1) / (self.period ** 2)) * self.scale).shape)
                image += torch.exp(-(self.order_vector - dpt[i*2:(i+1)*2]).pow(2).sum(dim=1) / (self.period ** 2)) * self.scale
            #     print(self.order_vector, dpt)
            # plt.imshow(pytorch_model.unwrap(image.view(self.order, self.order)))
            # plt.show()
            batch.append(image.view(self.order, self.order))
        im = torch.stack(batch, dim=0)
        return im.unsqueeze(1)

    def hidden(self, inputs, resp):
        x = self.create_image(inputs)
        return super().hidden(x)

    # def forward(self, inputs, resp):
    #     x = self.hidden(inputs)
    #     values, dist_entropy, probs, Q_vals = super().forward(x, resp)
    #     return values, dist_entropy, probs, Q_vals


