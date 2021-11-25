import os
import sys
import json
import pickle
import yaml
import re
import time
import random
import itertools
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


def convert_time(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    print(f"Total time: {h:02}:{m:02}:{s:02}")


def fix_random_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Multi-GPU
    if args.device == "multi":
        torch.cuda.manual_seed_all(args.seed)
    # Single-GPU
    else:
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = False  # If you want to set randomness, cudnn.benchmark = False
    cudnn.deterministic = True  # If you want to set randomness, cudnn.benchmark = True
    print(f"[Control randomness]\nseed: {args.seed}\n")


def str2list(string):
    if string == 'all':
        return 'all'
    else:
        return string.split(",")


def str2list_int(string):
    if string == 'all':
        return 'all'
    else:
        return list(map(int, string.split(",")))


def make_dir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)


def print_dict(dictionary):
    for key, value in dictionary.items():
        print(f"{key}: {value}")
    print("")


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def write_json(path, data, cls=MyEncoder):
    with open(path, "w") as json_file:
        json.dump(data, json_file, cls=cls)


def read_json(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data


def write_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def read_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def read_yaml(path):
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def print_off():
    sys.stdout = open(os.devnull, 'w')


def print_on():
    sys.stdout = sys.__stdout__


def one_hot_encoding(arr):
    num = len(np.unique(arr))
    encoding = np.eye(num)[arr]
    return encoding


def order_change(arr, order):
    arr = list(arr)
    tmp = arr[order[0]]
    arr[order[0]] = arr[order[1]]
    arr[order[1]] = tmp
    return arr


def transpose_tensor(tensor, order):
    return np.transpose(tensor, order_change(np.arange(len(tensor.shape)), order))


def import_model(model_name, config):
    module = importlib.import_module(f'models.{model_name}_model')
    model = getattr(module, config.name)(**config)
    return model


def init_weight(model, method):
    method = dict(normal=['normal_', dict(mean=0, std=0.01)],
                  xavier_uni=['xavier_uniform_', dict()],
                  xavier_normal=['xavier_normal_', dict()],
                  he_uni=['kaiming_uniform_', dict()],
                  he_normal=['kaiming_normal_', dict()]).get(method)
    if method is None:
        return None

    for module in model.modules():
        # LSTM
        if module.__class__.__name__ in ['LSTM']:
            for param in module._all_weights[0]:
                if param.startswith('weight'):
                    getattr(nn.init, method[0])(getattr(module, param), **method[1])
                elif param.startswith('bias'):
                    nn.init.constant_(getattr(module, param), 0)
        else:
            if hasattr(module, "weight"):
                # Not BN
                if not ("BatchNorm" in module.__class__.__name__):
                    getattr(nn.init, method[0])(module.weight, **method[1])
                # BN
                else:
                    nn.init.constant_(module.weight, 1)
                if hasattr(module, "bias"):
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)


def get_args(save_path):
    args = read_json(os.path.join(save_path, "args.json"))
    return args


def pretrained_model(save_path):
    args = get_args(save_path)
    print(f"S{args['subject']:02} is loaded.")
    try:
        model = read_pickle(os.path.join(save_path, 'model.pk'))
    except FileNotFoundError:
        raise FileNotFoundError
    save_path = set_pretrained_path(save_path)
    model = load_model(model, save_path)
    return model


def load_model(model, path, load_range='all'):
    checkpoint = torch.load(path, map_location='cpu')
    if next(iter(checkpoint['model_state_dict'].keys())).startswith('module'):
        new_state_dict = dict()
        for k, v in checkpoint['model_state_dict'].items():
            new_key = k[7:]
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict, strict=True)
    else:
        if load_range == 'all':
            target_module = set(map(lambda x: x.split(".")[0], checkpoint['model_state_dict'].keys()))
            print(f"Target module: {target_module}")
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            target_params = {k: v for k, v in checkpoint['model_state_dict'].items() if
                             k.split(".")[0] in load_range}
            model.load_state_dict(target_params, strict=False)
    return model


def set_pretrained_path(path):
    if not path.endswith('.tar'):
        tar = listdir_sort(os.path.join(path, 'checkpoints'))[-1]
        path = os.path.join(path, 'checkpoints', tar)
    return path


def listdir_sort(path):
    return sort(os.listdir(path))


def sort(array):
    '''
    sort exactly for list or array which element is string
    example: [1, 10, 2, 4] -> [1, 2, 4, 10]
    '''
    str2int = lambda string: int(string) if string.isdigit() else string
    key = lambda key: [str2int(x) for x in re.split("([0-9]+)", key)]
    return sorted(array, key=key)


def timeit(func):
    start = time.time()

    def decorator(*args):
        _return = func(*args)
        convert_time(time.time() - start)
        return _return

    return decorator


def guarantee_numpy(data):
    data_type = type(data)
    if data_type == torch.Tensor:
        device = data.device.type
        if device == 'cpu':
            data = data.detach().numpy()
        else:
            data = data.detach().cpu().numpy()
        return data
    elif data_type == np.ndarray or data_type == list:
        return data
    else:
        raise ValueError("Check your data type.")


def band_list(string):
    if string == 'all':
        return [[0, 4], [4, 7], [7, 13], [13, 30], [30, 42]]
    lst = string.split(",")
    assert len(lst) % 2 == 0, "Length of the list must be even number."
    it = iter(lst)
    return [list(map(int, itertools.islice(it, i))) for i in ([2] * (len(lst) // 2))]
