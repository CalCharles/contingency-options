import glob, os, torch
from ReinforcementLearning.models import pytorch_model

class MultiOption():
    def __init__(self, num_options=0, option_class=None): 
        self.option_index = 0
        self.num_options = num_options
        self.option_class = option_class
        self.models = []

    def initialize(self, args, num_options, state_class):
        self.models = []
        if args.model_form.find("dope") != -1:
            self.sess= tf.Session('', config=tf.ConfigProto(allow_soft_placement=True))
        else:
            self.sess = None
        for i in range(num_options):
            if not args.normalize:
                minmax = None
            else:
                minmax = state_class.get_minmax()
            model = self.option_class(args, state_class.flat_state_size() * args.num_stack, 
                state_class.action_num, factor=args.factor, name = args.unique_id + "_" + str(i) +"_", minmax = minmax, sess=self.sess)
            # since name is args.unique_id + str(i), this means that unique_id should be the edge, in form head_tail
            if args.cuda:
                model = model.cuda()
            self.models.append(model) # make this an argument controlled parameter
        self.option_index = 0

    def names(self):
        return [model.name for model in self.models]

    def determine_step(self, state, reward):
        '''
        output: 
        '''
        actions = []
        for i in range(self.num_options):
            actions.append(self.models[i](state, reward))
        pytorch_model.wrap(actions)
        return actions


    def determine_action(self, state):
        '''
        output: 4 x [batch_size, num_options, num_actions/1]
        '''
        values, dist_entropy, probs, Q_vals = [], [], [], []
        for i in range(self.num_options):
            v, de, p, Q = self.models[i](state)
            values.append(v)
            dist_entropy.append(de)
            probs.append(p)
            Q_vals.append(Q)
        return torch.stack(values, dim=0), torch.stack(dist_entropy, dim=0), torch.stack(probs, dim=0), torch.stack(Q_vals, dim=0)

    def layers(self, state):
        '''
        output: num_options x num layers x [batch, layer size]
        '''
        layers = []
        for i in range(self.num_options):
            layers.append(self.models[i].compute_layers(state))
        return layers


    def get_action(self, values, probs, Q_vals, index=-1):
        '''
        output 2 x [batch_size, num_actions]
        '''
        if index == -1:
            index = self.option_index
        return values[index], probs[index], Q_vals[index]

    def currentName(self):
        return self.models[self.option_index].name

    def currentModel(self):
        return self.models[self.option_index]

    def save(self, pth):
        for i, model in enumerate(self.models):
            model.save(pth)

    def load(self, args, pth):
        model_paths = glob.glob(os.path.join(pth, '*.pt'))
        model_paths.sort(key=lambda x: int(x.split("_")[1]))
        for mpth in model_paths:
            self.models.append(torch.load(mpth))
            if args.cuda:
                self.models[-1] = self.models[-1].cuda()
        self.option_index = 0
        self.num_options = len(self.models)
