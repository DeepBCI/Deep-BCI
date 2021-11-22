class Config(object):
    def __init__(self, args):
        self.lr = eval(args.lr)
        self.lr_str = args.lr
        self.num_epochs = len(self.lr)
        self.num_classes = 40
        self.num_subjects = 6
        self.num_splits = 6
        self.k = args.k
        self.batch_size_each = 200
        self.data_path = args.data_path
        self.model_path = args.model_path
        self.output_path = args.output_path
        self.target_subject = args.target_subject
        self.model_file = args.model_file
        self.data_file = args.data_file
        self.split_file = args.split_file
        self.split = args.split
        self.seed = args.seed

    def __str__(self):
        attrs = vars(self)
        attr_lst = sorted(attrs.keys())
        return '\n'.join("- %s: %s" % (item, attrs[item]) for item in attr_lst if item != 'lr')
