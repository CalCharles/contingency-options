class MultiOption():
    def __init__(self, num_options=0, option_class=None): 
        self.option_index = 0
        self.num_options = num_options
        self.option_class = option_class

    def initialize(self, args, num_options, state_class):
        self.models = []
        for i in range(num_options):
            model = self.option_class(args, state_class.flat_state_size * args.num_stack, 
                state_class.action_size, name = args.unique_id + "_" + str(i) +"_", minmax = state_class.get_minmax())
            # since name is args.unique_id + str(i), this means that unique_id should be the edge, in form head_tail
            if args.cuda:
                model = model.cuda()
            self.models.append(model) # make this an argument controlled parameter
        self.option_index = 0

    def names(self):
        return [model.name for model in self.models]

    def determine_action(self, state, index=-1):
        if index == -1:
            index = self.option_index
        return self.models[index](state)

    def currentName(self):
        return self.models[self.option_index].name

    def save(self, pth):
        for i, model in enumerate(self.models):
            torch.save(model, pth + model.name + ".pt")

    def load(self, args, pth):
        model_paths = glob.glob(os.path.join(pth, d, '*.pt'))
        model_paths.sort(key=lambda x: int(x.split("_")[1]))
        for mpth in model_paths:
            self.models.append(torch.load(mpth))
            if args.cuda:
                self.models[-1] = self.models[-1].cuda()
        self.option_index = 0
        self.num_options = len(self.models)
