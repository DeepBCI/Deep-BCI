import os
import torch
from torchinfo import summary

from utils.utils import import_model, pretrained_model, write_pickle


class ModelBuilder:
    def __init__(self, args):
        print("[Build Model]")
        self.model = self.__build_model(args)
        self.__set_device(self.model, args.device)
        self.model_summary(args, self.model)

    def __build_model(self, args):
        if args.mode == 'train':
            model = import_model(args.model, args.cfg)
            write_pickle(os.path.join(args.save_path, "model.pk"), model)
        else:
            model = pretrained_model(args.load_path)
        return model

    def __set_device(self, model, device):
        if device == 'cpu':
            device = torch.device("cpu")
        else:
            if not torch.cuda.is_available():
                raise ValueError("Check GPU")
            device = torch.device(f'cuda:{device}')
            torch.cuda.set_device(device)  # If you want to check device, use torch.cuda.current_device().
            model.cuda()
        # Print device
        model.device = device
        print(f"device: {device}")
        print("")

    def model_summary(self, args, model):
        if args.summary:
            results = summary(model, args.cfg.input_shape, col_names=["kernel_size", "output_size", "num_params"],
                              device=model.device if not model.device == 'multi' else torch.device("cuda:0"))
            args.trainable_params = results.trainable_params
            print("")