import random
import numpy as np
import yaml
import os
import torch


def set_seed(seed):
    """
    random seed for generation
    :param seed: manual seed from config.yaml
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



def get_config(config):
    """
    get configuration from config.yaml
    :param config: config.yaml
    :return: Dict(Key, Value(list))
    """
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)



def ensure_dir(path):
    """
    check the existence of path, before creating it
    :param path: path
    :return:
    """
    if not os.path.exists(path):
        print("creating folder", path)
        os.makedirs(path)
    else:
        print(path, "path already exist")



def ensure_dirs(paths):
    """
    multiple-path version of ensuer_dir
    :param paths: list of path
    :return:
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            ensure_dir(path)
    else:
        ensure_dir(paths)



def print_composite(data, beg=""):
    """
    print the elements of composite
    for dict, list, np.ndarray, torch.Tensor
    :param data: composite
    :param beg: name of composite
    :return:
    """
    if isinstance(data, dict):
        print(f'{beg} dict, size={len(data)}')
        for key, value in data.items():
            print(f'    {beg}{key}:')
            print_composite(value, beg + "    ")
    elif isinstance(data, list):
        print(f'{beg} list, len={len(data)}')
        for i, item in enumerate(data):
            print(f'    {beg}item {i}')
            print_composite(item, beg+ "    ")
    elif isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
        print(f'{beg} array of size {data.shape}')
    else:
        print(f'{beg} {data}')

def gen_model_list(dirname, key):
    """
    generate model list from given directory
    :param dirname: directory that includes .pt
    :param key:
    :return: joined path (path + filename)
    """
    if os.path.exists(dirname) is False:
        print(f'expected directory name {dirname} is not exist!')
        return None
    gen_models=[os.path.join(dirname, f) for f in os.listdir(dirname) if
                os.path.isfile(os.path.join(dirname, f)) and
                key in f and ".pt" in f]
    if gen_models is None or len(gen_models) == 0:
        print(f'gen_model fails!. please check {dirname} includes .pt files')
        return None
    gen_models.sort()
    last_model_name=gen_models[-1]
    return last_model_name


