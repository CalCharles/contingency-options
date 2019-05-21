import glob, os, torch, copy
import numpy as np
from Models.models import pytorch_model, models

# parameterized options argument: 0: no parametrization, 1: discrete parameters, 2: continuous parameters, 3: combination of discrete and continuous (not implemented)


class MultiOption():
    def __init__(self, num_options=0, option_class=None):
        self.option_index = 0
        self.num_options = num_options
        self.option_class = option_class
        self.models = []

    def session(self, args):
        if args.model_form.find("dope") != -1:
            self.sess= tf.Session('', config=tf.ConfigProto(allow_soft_placement=True))
        else:
            self.sess = None


    def create_model(self, args, state_class, parameter_minmax, i, num_actions, model_class=None, parameter=1):
        if not args.normalize:
            minmax = None
        else:
            minmax = state_class.get_minmax()
        state_dim = state_class.flat_state_size() * args.num_stack
        if args.parameterized_option == 2:
            state_dim = state_dim + parameter_minmax[0].shape[0]
            if args.normalize:
                minmax = (np.hstack((minmax[0], parameter_minmax[0])) , np.hstack((minmax[1], parameter_minmax[1])))
        if model_class is None:
            print(self.option_class)
            model = self.option_class(args=args, num_inputs=state_class.flat_state_size() * args.num_stack, num_outputs=num_actions, 
                class_sizes = state_class.sizes, factor=args.factor, name = args.unique_id + "__" + str(i) +"__", minmax = minmax, sess=self.sess, param_dim=parameter)
        else:
            model = model_class(args=args, num_inputs=state_class.flat_state_size() * args.num_stack, num_outputs=num_actions, 
                class_sizes = state_class.sizes, factor=args.factor, name = args.unique_id + "__" + str(i) +"__", minmax = minmax, sess=self.sess, param_dim=parameter)
        return model

    def cuda():
        for i in range(len(self.models)):
            self.models[i] = model.cuda()
        return self

    def cpu():
        for i in range(len(self.models)):
            self.models[i] = model.cpu()
        return self

    def train(self):
        for model in self.models:
            model.train()

    def initialize(self, args, num_options, state_class, num_actions, parameter_minmax = None):
        '''
        Naming for models is based on double underscore __
        parameterized options argument: 0: no parametrization, 1: discrete parameters, 2: continuous parameters, 3: combination of discrete and continuous (not implemented)
        '''
        self.iscuda = False
        self.session(args)
        if not args.load_weights:
            self.models = []
            if args.parameterized_option > 0:
                param = num_options
                if args.parameterized_option == 2:
                    param = parameter_minmax[0].shape[0]
                model = self.create_model(args, state_class, parameter_minmax, 0, num_actions, parameter=param)
                # since name is args.unique_id + str(i), this means that unique_id should be the edge, in form head_tail
                if args.cuda:
                    model = model.cuda()
                self.models.append(model) # make this an argument controlled parameter
            else:
                for i in range(num_options):
                    model = self.create_model(args, state_class, parameter_minmax, i, num_actions)
                    # since name is args.unique_id + str(i), this means that unique_id should be the edge, in form head_tail
                    if args.cuda:
                        model = model.cuda()
                    self.models.append(model) # make this an argument controlled parameter
        self.num_options = num_options
        self.parameterized_option = args.parameterized_option
        self.parameter_dim = 2
        if args.parameterized_option == 2:
            self.parameter_dim = parameter_minmax[0].shape[0]

        self.option_index = 0
        print("parameterized option", self.parameterized_option)

    def names(self):
        return [model.name for model in self.models]

    def determine_step(self, state, reward):
        '''
        output: what is this function?
        '''
        actions = []
        for i in range(self.num_options):
            actions.append(self.models[i](state, reward))
        pytorch_model.wrap(actions)
        return actions

    def set_parameter(self, parameter):
        if self.parameterized_option == 2:
            self.models[0].option_input = parameter

    def determine_action(self, state, resp):
        '''
        output: 4 x [batch_size, num_options, num_actions/1]
        '''
        values, log_probs, probs, Q_vals = [], [], [], []
        for i in range(self.num_options):
            if self.parameterized_option == 1: # TODO: parametrized options only has one option (possibly a combination)
                self.models[0].option_index = i
            v, lp, p, Q = self.models[i](state, resp)
            values.append(v)
            log_probs.append(lp)
            probs.append(p)
            Q_vals.append(Q)
        return torch.stack(values, dim=0), torch.stack(log_probs, dim=0), torch.stack(probs, dim=0), torch.stack(Q_vals, dim=0)

    def layers(self, state):
        '''
        output: num_options x num layers x [batch, layer size]
        '''
        layers = []
        for i in range(self.num_options):
            layers.append(self.models[i].compute_layers(state))
        return layers


    def get_action(self, values, probs, log_probs, Q_vals, index=-1):
        '''
        output 3 x [num_option, batch_size, num_actions]
        '''

        if index == -1:
            index = self.option_index
        return values[index], probs[index], log_probs[index], Q_vals[index]

    def currentName(self):
        return self.models[self.option_index].name

    def currentModel(self):
        return self.models[self.option_index]

    def currentOptionParam(self):
        return self.models[self.option_index].option_values     

    def save(self, pth):
        print(os.path.join(*pth.split("/")))
        try:
            os.makedirs(os.path.join(*pth.split("/")))
        except OSError:
            pass
        for i, model in enumerate(self.models):
            model.save(pth)

    def load(self, args, pth):
        model_paths = glob.glob(os.path.join(pth, '*.pt'))
        print("loading", model_paths)
        model_paths.sort(key=lambda x: int(x.split("__")[1]))
        for mpth in model_paths:
            loaded_model = torch.load(mpth, map_location='cpu').cuda()
            try: # use the mean of a population model if we are not training a population model
                pop = loaded_model.num_population > 0
            except AttributeError as e:
                pop = False
            if pop and args.model_form != 'population':
                loaded_model = loaded_model.mean
            self.models.append(loaded_model)
            if args.cuda:
                self.models[-1] = self.models[-1].cuda()
            self.models[-1].test=True
        self.option_index = 0
        self.num_options = len(self.models)
        if len(self.models) > 0:
            print(self.models)
            self.parameterized_option = self.models[0].parameterized_option
            self.parameter_dim = self.models[0].param_dim

    def duplicate(self, num_models, args, state_class, num_actions, parameter_minmax):
        # TODO: only handles 1->many of models, build in many->many handling
        old_model = self.models[0]
        self.models = []
        param = 1
        if args.parameterized_option == 1:
            param = num_models
        elif args.parameterized_option == 2:
            param = parameter_minmax[0].shape[0]
        for i in range(num_models):
            print(args.model_form, type(old_model) != models['population'])
            if args.model_form == 'population' and type(old_model) != models['population']:
                # if we are switching to a population model, copy the new model into the mean and the values
                new_model = self.create_model(args, state_class, parameter_minmax, i, num_actions, model_class=models[args.model_form], parameter=param)
                for j in range(args.num_population):
                    new_model.networks[j] = copy.deepcopy(old_model)
                new_model.mean = copy.deepcopy(old_model)
                new_model.best = copy.deepcopy(old_model)
            else:
                new_model = copy.deepcopy(old_model)
            new_model.test = not args.train
            new_model.name = args.unique_id + "__" + str(i) +"__"
            if args.model_form == 'population' and args.base_form == 'adjust': # right now, basically the same logic but for each model in the population
                print("param", param)
                networks = []
                for j in range(new_model.num_population):
                    m = self.create_model(args, state_class, parameter_minmax, i, num_actions, models[args.base_form], parameter = param)
                    m.load_base(new_model.networks[j])
                    networks.append(m)
                new_model.set_networks(networks)
                mean = self.create_model(args, state_class, parameter_minmax, i, num_actions, models[args.base_form], parameter = param)
                mean.load_base(new_model.mean)
                new_model.mean = mean
                best = self.create_model(args, state_class, parameter_minmax, i, num_actions, models[args.base_form], parameter = param)
                best.load_base(new_model.best)
                new_model.best = best
                if args.cuda:
                    new_model = new_model.cuda()
                self.models.append(new_model)
            elif args.model_form == 'adjust': # for now, while there is only one adjustment model
                print("loading adjustment model")
                model = self.create_model(args, state_class, parameter_minmax, i, num_actions, models[args.model_form], parameter = param)
                model.load_base(new_model)
                if args.cuda:
                    model = model.cuda()
                self.models.append(model)
            else:
                if args.cuda:
                    new_model = new_model.cuda()
                self.models.append(new_model)
        self.num_options = num_models
        self.option_index = 0
